import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
import os
import sys
from pathlib import Path
from wiki_loader import WikipediaDataSet
import accuracy
from models import naive
from timeit import default_timer as timer


logger = utils.setup_logger(__name__, 'test_accuracy.log')


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums

def get_segments_folders(path):
    ret_folders = []
    folders = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    for folder in folders:
        if '-' in folder:
            ret_folders.append(os.path.join(path, folder))
    return ret_folders


def main(args):
    start = timer()

    sys.path.append(str(Path(__file__).parent))

    utils.read_config_file(args.config)
    utils.config.update(vars(args))  # Use vars for better argument handling

    logger.debug('Running with config %s', utils.config)
    print(f'Running with threshold: {args.seg_threshold}')
    preds_stats = utils.predictions_analysis()

    if not args.test:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    else:
        word2vec = None

    word2vec_done = timer()
    print(f'Loading word2vec elapsed: {word2vec_done - start} seconds')
    dirname = 'test'

    # Determine dataset based on wiki flag
    if args.wiki:
        dataset_folders = [Path(utils.config['wikidataset']) / dirname]
        if args.wiki_folder:
            dataset_folders = [args.wiki_folder]
        print('Running on Wikipedia')
    else:
        if args.bySegLength:
            dataset_folders = get_segments_folders(utils.config['choidataset'])
            print('Running on Choi by segments length')
        else:
            dataset_folders = [utils.config['choidataset']]
            print('Running on Choi')

    # Load the model
    with open(args.model, 'rb') as f:
        model = torch.load(f)

    model = maybe_cuda(model)
    model.eval()

    if args.naive:
        model = naive.create()

    for dataset_path in dataset_folders:
        if args.bySegLength:
            print('Segment is', os.path.basename(dataset_path), ":")

        # Load dataset
        if args.wiki:
            if args.wiki_folder:
                dataset = WikipediaDataSet(dataset_path, word2vec, folder=True, high_granularity=False)
            else:
                dataset = WikipediaDataSet(dataset_path, word2vec, high_granularity=False)
        else:
            dataset = ChoiDataset(dataset_path, word2vec)

        dl = DataLoader(dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=False)

        # Testing loop
        with tqdm(desc='Testing', total=len(dl)) as pbar:
            total_accurate = 0
            total_count = 0
            total_loss = 0
            acc = accuracy.Accuracy()

            for i, (data, targets, paths) in enumerate(dl):
                if i == args.stop_after:
                    break

                pbar.update()
                output = model(data)
                targets_var = maybe_cuda(torch.cat(targets, 0), args.cuda)
                batch_loss = 0
                output_prob = softmax(output.cpu().numpy())
                output_seg = output_prob[:, 1] > args.seg_threshold
                target_seg = targets_var.cpu().numpy()
                batch_accurate = (output_seg == target_seg).sum()
                total_accurate += batch_accurate
                total_count += len(target_seg)
                total_loss += batch_loss
                preds_stats.add(output_seg, target_seg)

                current_target_idx = 0
                for k, t in enumerate(targets):
                    document_sentence_count = len(t)
                    sentences_length = [s.size()[0] for s in data[k]] if args.calc_word else None
                    to_idx = int(current_target_idx + document_sentence_count)
                    h = output_seg[current_target_idx: to_idx]

                    # Add classification for the last sentence
                    h = np.append(h, [1])
                    t = np.append(t.cpu().numpy(), [1])

                    acc.update(h, t, sentences_length=sentences_length)

                    current_target_idx = to_idx

                logger.debug('Batch %s - error %7.4f, Accuracy: %7.4f', i, batch_loss, batch_accurate / len(target_seg))
                pbar.set_description('Testing, Accuracy={:.4}'.format(batch_accurate / len(target_seg)))

        # Logging results
        average_loss = total_loss / len(dl)
        average_accuracy = total_accurate / total_count
        calculated_pk, _ = acc.calc_accuracy()

        logger.info('Finished testing.')
        logger.info('Average loss: %s', average_loss)
        logger.info('Average accuracy: %s', average_accuracy)
        logger.info(f'Pk: {calculated_pk:.4}.')
        logger.info(f'F1: {preds_stats.get_f1():.4}.')

        end = timer()
        print(f'Seconds to execute the whole flow: {end - start}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g., fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--model', help='Model to run', required=True)
    parser.add_argument('--stop_after', help='Number of batches to stop after', type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use Wikipedia as dataset?', action='store_true')
    parser.add_argument('--bySegLength', help='Calculate pk on Choi by segments length?', action='store_true')
    parser.add_argument('--wiki_folder', help='Path to folder containing wiki documents')
    parser.add_argument('--naive', help='Use naive model', action='store_true')
    parser.add_argument('--seg_threshold', help='Threshold for binary classification', type=float, default=0.4)
    parser.add_argument('--calc_word', help='Calculate P_K by word', action='store_true')

    main(parser.parse_args())