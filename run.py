import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.nn import DataParallel
from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
from tensorboard_logger import configure, log_value
import time
import os
import sys
from pathlib import Path
from wiki_loader import WikipediaDataSet
import accuracy
import numpy as np
from termcolor import colored
from models.max_sentence_embedding import Model

# torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(42)

preds_stats = utils.predictions_analysis()

def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums

def import_model(model_name):
    module = __import__('models.' + model_name, fromlist=['models'])
    return module.create()

class Accuracies:
    def __init__(self):
        self.thresholds = np.arange(0, 1, 0.05)
        self.accuracies = {k: accuracy.Accuracy() for k in self.thresholds}

    def update(self, output_np, targets_np):
        current_idx = 0
        for k, t in enumerate(targets_np):
            document_sentence_count = len(t)
            to_idx = int(current_idx + document_sentence_count)

            for threshold in self.thresholds:
                output = (output_np[current_idx: to_idx, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])

                self.accuracies[threshold].update(h, tt)

            current_idx = to_idx

    def calc_accuracy(self):
        min_pk = np.inf
        min_threshold = None
        min_epoch_windiff = None
        for threshold in self.thresholds:
            epoch_pk, epoch_windiff = self.accuracies[threshold].calc_accuracy()
            if epoch_pk < min_pk:
                min_pk = epoch_pk
                min_threshold = threshold
                min_epoch_windiff = epoch_windiff

        return min_pk, min_epoch_windiff, min_threshold

def tensor_size_in_bytes(tensor):
    return tensor.numel() * tensor.element_size()

def compute_batch_size(data):
    total_size=0

    for element in data:
        num_sentences = len(element)
        
        for sentence in element:
            total_size += tensor_size_in_bytes(sentence)

    return total_size / (1024**2)



def train(model, args, epoch, dataset, logger, optimizer):
    model.train()
    total_loss = 0.0  # Changed to float value
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, target, paths) in enumerate(dataset):
            if i == args.stop_after:
                break
            pbar.update()
            model.zero_grad()

            try:
                output = model(data)
            except Exception as e:
                print(f"Error while passing batch {i+1} to the model")
                print(f"Exception: {e}")
                print(f"Paths: {paths}")
                continue
            
            target_var = maybe_cuda(torch.cat(target, 0), args.cuda)
            loss = model.criterion(output, target_var)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()  # Replaced deprecated .data[0] with .item()

            pbar.set_description(f'Training, loss={loss.item():.4}')

    total_loss /= len(dataset)
    logger.debug(f'Training Epoch: {epoch + 1}, Loss: {total_loss:.4}')
    # log_value('Training Loss', total_loss, epoch + 1)

def validate(model, args, epoch, dataset, logger):
    model.eval()
    with tqdm(desc='Validating', total=len(dataset)) as pbar:
        acc = Accuracies()
        for i, (data, target, paths) in enumerate(dataset):
            if i == args.stop_after:
                break
            pbar.update()
            
            try:
                output = model(data)
                output_softmax = F.softmax(output, dim=1)
            except Exception as e:
                print(f"Error while passing batch {i+1} to the model")
                print(f"Exception: {e}")
                print(f"Paths: {paths}")
                continue            
            
            targets_var = maybe_cuda(torch.cat(target, 0), args.cuda)

            output_seg = output.argmax(dim=1).cpu().numpy()
            target_seg = targets_var.cpu().numpy()
            preds_stats.add(output_seg, target_seg)

            acc.update(output_softmax.detach().cpu().numpy(), target)

        epoch_pk, epoch_windiff, threshold = acc.calc_accuracy()

        logger.info(f'Validating Epoch: {epoch + 1}, accuracy: {preds_stats.get_accuracy():.4}, '
                    f'Pk: {epoch_pk:.4}, Windiff: {epoch_windiff:.4}, F1: {preds_stats.get_f1():.4}')
        preds_stats.reset()

        return epoch_pk, threshold

def test(model, args, epoch, dataset, logger, threshold):
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        acc = accuracy.Accuracy()
        for i, (data, target, paths) in enumerate(dataset):
            if i == args.stop_after:
                break
            pbar.update()
            
            try:
                output = model(data)
                output_softmax = F.softmax(output, dim=1)
            except Exception as e:
                print(f"Error while passing batch {i+1} to the model")
                print(f"Exception: {e}")
                print(f"Paths: {paths}")
                continue
            
            targets_var = maybe_cuda(torch.cat(target, 0), args.cuda)
            output_seg = output.argmax(dim=1).cpu().numpy()
            target_seg = targets_var.cpu().numpy()
            preds_stats.add(output_seg, target_seg)

            current_idx = 0
            for k, t in enumerate(target):
                document_sentence_count = len(t)
                to_idx = int(current_idx + document_sentence_count)

                output = (output_softmax.detach().cpu().numpy()[current_idx:to_idx, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])

                acc.update(h, tt)
                current_idx = to_idx

        epoch_pk, epoch_windiff = acc.calc_accuracy()

        logger.debug(f'Testing Epoch: {epoch + 1}, accuracy: {preds_stats.get_accuracy():.4}, '
                     f'Pk: {epoch_pk:.4}, Windiff: {epoch_windiff:.4}, F1: {preds_stats.get_f1():.4}')
        preds_stats.reset()

        return epoch_pk


def load_model_and_optimizer(checkpoint_path, is_cuda, model, optimizer):

    map_location = torch.device('cuda') if is_cuda else torch.device('cpu')

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded model and optimizer state from {checkpoint_path}")
    return model, optimizer


def main(args):
    sys.path.append(str(Path(__file__).parent))

    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))

    utils.read_config_file(args.config)
    utils.config.update(vars(args))  # Updated to use vars(args)
    logger.debug(f'Running with config {utils.config}')

    word2vec = None if args.test else gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)

    if not args.infer:
        dataset_class = WikipediaDataSet if args.wiki else ChoiDataset
        dataset_path = Path(utils.config['wikidataset']) if args.wiki else Path(utils.config['choidataset'])

        train_dataset = dataset_class(dataset_path / 'train', word2vec, high_granularity=args.high_granularity)
        dev_dataset = dataset_class(dataset_path / 'dev', word2vec, high_granularity=args.high_granularity)
        test_dataset = dataset_class(dataset_path / 'test', word2vec, high_granularity=args.high_granularity)

        if args.subset:
            train_dataset = Subset(train_dataset,range(1000))
            dev_dataset = Subset(dev_dataset,range(1000))
            test_dataset = Subset(test_dataset,range(1000))

        train_batch_size,test_batch_size = args.bs, args.test_bs
        
        if torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()
            print(f"Using {num_gpus} GPUs")
            train_batch_size = args.bs * num_gpus
            test_batch_size = args.test_bs * num_gpus
        
        train_dl = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True,
                              num_workers=args.num_workers,pin_memory=args.pin_memory)
        dev_dl = DataLoader(dev_dataset, batch_size=test_batch_size, collate_fn=collate_fn, shuffle=False,
                            num_workers=args.num_workers,pin_memory=args.pin_memory)
        test_dl = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers,pin_memory=args.pin_memory)

    model = Model(input_size=300, hidden=256, num_layers=2)
    model = maybe_cuda(model)

    if torch.cuda.device_count() > 1 and not args.infer:
        model = DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.load_from:
        model, optimizer = load_model_and_optimizer(args.load_from, args.cuda, model, optimizer)

    if args.benchmark:
        for j in range(args.epochs):
            train(model, args, j, train_dl, logger, optimizer)
        return 

    if not args.infer:
        best_val_pk = 1.0
        for j in range(args.epochs):
            train(model, args, j, train_dl, logger, optimizer)
            if isinstance(model, torch.nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save({
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict()
            }, open(checkpoint_path / f'model{j:03d}.pt', 'wb'))

            val_pk, threshold = validate(model, args, j, dev_dl, logger)
            print(f'Current best model from epoch {j} with p_k {val_pk} and threshold {threshold}')
            if val_pk < best_val_pk:
                test_pk = test(model, args, j, test_dl, logger, threshold)
                print(f'Current best model from epoch {j} with p_k {test_pk} and threshold {threshold}')
                best_val_pk = val_pk
                if isinstance(model, torch.nn.DataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                torch.save({
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                }, open(checkpoint_path / f'best_model.pt', 'wb'))

    else:
        test_dl = DataLoader(WikipediaDataSet(args.infer, word2vec=word2vec, high_granularity=args.high_granularity),
                             batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False, num_workers=args.num_workers,pin_memory=args.pin_memory)
        print(test(model, args, 0, test_dl, logger, 0.4))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--pin_memory', help='Pin Memory?', action='store_true')
    parser.add_argument('--subset', help='Use a sample of 1000 rows', action='store_true')
    parser.add_argument('--benchmark', help='Use PyTorch profiler', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g. fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--test_bs', help='Test batch size', type=int, default=5)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--model', help='Model to run - will import and run',default='max_sentence_embedding')
    parser.add_argument('--load_from', help='Location of a .pt model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use Wikipedia as dataset?', action='store_true')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--high_granularity', help='Use high granularity for Wikipedia dataset segmentation', action='store_true')
    parser.add_argument('--infer', help='Inference directory', type=str)

    main(parser.parse_args())