import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
from tensorboard_logger import configure, log_value
import os
import sys
from pathlib import Path
from wiki_loader import WikipediaDataSet
import accuracy
import numpy as np
from termcolor import colored

torch.multiprocessing.set_sharing_strategy('file_system')

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

def train(model, args, epoch, dataset, logger, optimizer):
    model.train()
    total_loss = 0.0  # Changed to float value
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, target, paths) in enumerate(dataset):
            if i == args.stop_after:
                break

            pbar.update()
            model.zero_grad()
            output = model(data)
            target_var = maybe_cuda(torch.cat(target, 0), args.cuda)
            loss = model.criterion(output, target_var)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()  # Replaced deprecated .data[0] with .item()

            pbar.set_description(f'Training, loss={loss.item():.4}')

    total_loss /= len(dataset)
    logger.debug(f'Training Epoch: {epoch + 1}, Loss: {total_loss:.4}')
    log_value('Training Loss', total_loss, epoch + 1)

def validate(model, args, epoch, dataset, logger):
    model.eval()
    with tqdm(desc='Validating', total=len(dataset)) as pbar:
        acc = Accuracies()
        for i, (data, target, paths) in enumerate(dataset):
            if i == args.stop_after:
                break
            pbar.update()
            output = model(data)
            output_softmax = F.softmax(output, dim=1)
            targets_var = maybe_cuda(torch.cat(target, 0), args.cuda)

            output_seg = output.argmax(dim=1).cpu().numpy()
            target_seg = targets_var.cpu().numpy()
            preds_stats.add(output_seg, target_seg)

            acc.update(output_softmax.cpu().numpy(), target)

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
            output = model(data)
            output_softmax = F.softmax(output, dim=1)
            targets_var = maybe_cuda(torch.cat(target, 0), args.cuda)
            output_seg = output.argmax(dim=1).cpu().numpy()
            target_seg = targets_var.cpu().numpy()
            preds_stats.add(output_seg, target_seg)

            current_idx = 0
            for k, t in enumerate(target):
                document_sentence_count = len(t)
                to_idx = int(current_idx + document_sentence_count)

                output = (output_softmax.cpu().numpy()[current_idx:to_idx, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])

                acc.update(h, tt)
                current_idx = to_idx

        epoch_pk, epoch_windiff = acc.calc_accuracy()

        logger.debug(f'Testing Epoch: {epoch + 1}, accuracy: {preds_stats.get_accuracy():.4}, '
                     f'Pk: {epoch_pk:.4}, Windiff: {epoch_windiff:.4}, F1: {preds_stats.get_f1():.4}')
        preds_stats.reset()

        return epoch_pk

def main(args):
    sys.path.append(str(Path(__file__).parent))

    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))

    utils.read_config_file(args.config)
    utils.config.update(vars(args))  # Updated to use vars(args)
    logger.debug(f'Running with config {utils.config}')

    configure(os.path.join('runs', args.expname))

    word2vec = None if args.test else gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)

    if not args.infer:
        dataset_class = WikipediaDataSet if args.wiki else ChoiDataset
        dataset_path = Path(utils.config['wikidataset']) if args.wiki else Path(utils.config['choidataset'])

        train_dataset = dataset_class(dataset_path / 'train', word2vec, high_granularity=args.high_granularity)
        dev_dataset = dataset_class(dataset_path / 'dev', word2vec, high_granularity=args.high_granularity)
        test_dataset = dataset_class(dataset_path / 'test', word2vec, high_granularity=args.high_granularity)

        train_dl = DataLoader(train_dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=True,
                              num_workers=args.num_workers,pin_memory=args.pin_memory)
        dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                            num_workers=args.num_workers,pin_memory=args.pin_memory)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers,pin_memory=args.pin_memory)

    model = import_model(args.model) if args.model else torch.load(open(args.load_from, 'rb'))
    model = maybe_cuda(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if not args.infer:
        best_val_pk = 1.0
        for j in range(args.epochs):
            train(model, args, j, train_dl, logger, optimizer)
            torch.save(model, open(checkpoint_path / f'model{j:03d}.pt', 'wb'))

            val_pk, threshold = validate(model, args, j, dev_dl, logger)
            if val_pk < best_val_pk:
                test_pk = test(model, args, j, test_dl, logger, threshold)
                logger.debug(colored(f'Current best model from epoch {j} with p_k {test_pk} and threshold {threshold}', 'green'))
                best_val_pk = val_pk
                torch.save(model, open(checkpoint_path / 'best_model.pt', 'wb'))

    else:
        test_dl = DataLoader(WikipediaDataSet(args.infer, word2vec=word2vec, high_granularity=args.high_granularity),
                             batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False, num_workers=args.num_workers,pin_memory=args.pin_memory)
        print(test(model, args, 0, test_dl, logger, 0.4))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--pin_memory', help='Pin Memory?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g. fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--test_bs', help='Test batch size', type=int, default=5)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--model', help='Model to run - will import and run')
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use Wikipedia as dataset?', action='store_true')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--high_granularity', help='Use high granularity for Wikipedia dataset segmentation', action='store_true')
    parser.add_argument('--infer', help='Inference directory', type=str)

    main(parser.parse_args())