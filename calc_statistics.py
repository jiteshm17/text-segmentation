import torch
from torch.utils.data import DataLoader
import numpy as np

from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import utils
import sys
from pathlib import Path  # Use pathlib instead of pathlib2
from wiki_loader import WikipediaDataSet
import accuracy

logger = utils.setup_logger(__name__, 'train.log')

def main(args):
    sys.path.append(str(Path(__file__).parent))

    utils.read_config_file(args.config)
    utils.config.update(vars(args))  # Update config with args dictionary

    logger.debug('Running with config %s', utils.config)
    article_with_problems = 0

    dataset = WikipediaDataSet("dataset_path", word2vec=None,
                               folder=True, high_granularity=False)

    num_sentences = 0
    num_segments = 0
    num_documents = 0
    min_num_segment = 1000
    max_num_segment = 0
    min_num_sentences = 1000
    max_num_sentences = 0

    dl = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    docs_num_segments_vec = np.zeros(len(dl))
    segments_num_sentences_vec = []
    print(f'Number of documents: {len(dl)}')

    with tqdm(desc='Testing', total=len(dl)) as pbar:
        for i, (data, targets, paths) in enumerate(dl):
            if len(paths) == 0:
                article_with_problems += 1
                docs_num_segments_vec[i] = np.nan
                continue
            try:
                if i % 1000 == 0 and i > 0:
                    print(i)
                if len(targets) > 0:
                    targets_var = maybe_cuda(torch.cat(targets, 0), None)
                    target_seg = targets_var.cpu().numpy()
                    target_seg = np.concatenate([target_seg, np.array([1])])
                else:
                    target_seg = np.ones(1)
                
                num_sentences += len(target_seg)
                doc_num_of_segment = sum(target_seg)
                
                min_num_segment = min(min_num_segment, doc_num_of_segment)
                max_num_segment = max(max_num_segment, doc_num_of_segment)
                
                num_segments += doc_num_of_segment
                num_documents += 1
                docs_num_segments_vec[i] = doc_num_of_segment

                one_inds = np.where(target_seg == 1)[0]
                one_inds += 1
                one_inds = np.concatenate(([0], one_inds))

                if len(one_inds) == 1:
                    sentences_in_segments = [len(target_seg)]
                else:
                    sentences_in_segments = one_inds[1:] - one_inds[:-1]
                
                segments_num_sentences_vec = np.concatenate((segments_num_sentences_vec, sentences_in_segments))
                current_min = np.min(sentences_in_segments)
                current_max = np.max(sentences_in_segments)
                
                min_num_sentences = min(min_num_sentences, current_min)
                max_num_sentences = max(max_num_sentences, current_max)

            except Exception as e:
                logger.info(f'Exception "{e}" in batch {i}')
                logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
                raise

    print(f'Total sentences: {num_sentences}.')
    print(f'Total segments: {num_segments}.')
    print(f'Total documents: {num_documents}.')
    print(f'Average segment size: {num_sentences / num_segments:.3f}.')
    print(f'Min #segments in a document: {min_num_segment}.')
    print(f'Max #segments in a document: {max_num_segment}.')
    print(f'Min #sentences in a segment: {min_num_sentences}.')
    print(f'Max #sentences in a segment: {max_num_sentences}.')

    print('\nNew computing method\n')
    print(f'Number of documents: {len(docs_num_segments_vec) - np.isnan(docs_num_segments_vec).sum()}.')
    print(f'Total segments: {np.nansum(docs_num_segments_vec)}.')
    print(f'Total sentences: {np.sum(segments_num_sentences_vec)}.')

    print(f'Min #segments in a document: {np.nanmin(docs_num_segments_vec)}.')
    print(f'Max #segments in a document: {np.nanmax(docs_num_segments_vec)}.')
    print(f'Mean segments in a document: {np.nanmean(docs_num_segments_vec):.3f}.')
    print(f'Standard deviation of segments in a document: {np.nanstd(docs_num_segments_vec):.3f}.')

    print(f'\nMin #sentences in a segment: {np.min(segments_num_sentences_vec)}.')
    print(f'Max #sentences in a segment: {np.max(segments_num_sentences_vec)}.')
    print(f'Average segment size: {np.mean(segments_num_sentences_vec):.3f}.')
    print(f'Standard deviation of segment size: {np.std(segments_num_sentences_vec):.3f}.')

    print(f'\nArticles with problems: {article_with_problems}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    main(parser.parse_args())