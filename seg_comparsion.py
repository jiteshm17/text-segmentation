from argparse import ArgumentParser
import utils
from utils import maybe_cuda
import gensim
import torch
from torch.autograd import Variable
from test_accuracy_manifesto import ManifestoDataset
from wiki_loader import WikipediaDataSet
from choiloader import ChoiDataset, collate_fn, read_choi_file
from torch.utils.data import DataLoader
from test_accuracy import softmax
from wiki_loader import clean_section, split_sentences, section_delimiter, extract_sentence_words
import os
import sys

preds_stats = utils.predictions_analysis()
paragraphs_delimiter = "=="

def main(args):
    utils.read_config_file(args.config)

    if not args.test:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    else:
        word2vec = None

    # Load model
    with open(args.model, 'rb') as f:
        model = torch.load(f)
    model = maybe_cuda(model)
    model.eval()

    # Set dataset and delimiter based on the input type
    if args.wiki:
        dataset = WikipediaDataSet(args.folder, word2vec, folder=True)
        delimiter = section_delimiter
    elif args.choi:  # Not in use but kept for reference
        dataset = ChoiDataset(args.folder, word2vec, is_cache_path=True)
        delimiter = paragraphs_delimiter
    else:
        print('Dataset type is required')
        return

    dl = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    # Process each document in the dataset
    for i, (data, targets, paths) in enumerate(dl):
        doc_path = str(paths[0])
        output = model(data)
        targets_var = Variable(maybe_cuda(torch.cat(targets, 0), args.cuda), requires_grad=False)

        output_prob = softmax(output.data.cpu().numpy())
        output_seg = output_prob[:, 1] > 0.3
        target_seg = targets_var.data.cpu().numpy()
        preds_stats.add(output_seg, target_seg)

        # Create the output folder if it doesn't exist
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

        # Write the result file
        result_file_path = os.path.join(args.output_folder, os.path.basename(doc_path))
        with open(result_file_path, "w", encoding='utf-8') as result_file:
            with open(doc_path, "r", encoding='utf-8') as file:
                raw_content = file.read()

            sections = [clean_section(s) for s in raw_content.strip().split(delimiter) if len(s) > 0 and s != "\n"]

            sum_sentences = 0
            total_num_sentences = 0
            bad_sentences = 0

            for section in sections:
                sentences = split_sentences(section)
                if sentences:
                    total_num_sentences += len(sentences)
                    for i, sentence in enumerate(sentences):
                        words = extract_sentence_words(sentence)
                        sentence = " ".join(words)

                        result_file.write(sentence + "\n")

                        if len(target_seg) == sum_sentences:  # Last sentence
                            continue

                        if target_seg[sum_sentences]:  # True segmentation
                            result_file.write(delimiter + "\n")
                        
                        if output_seg[sum_sentences]:  # Model segmentation
                            result_file.write("*******Our_Segmentation********\n")
                        
                        sum_sentences += 1

        if (total_num_sentences - bad_sentences) != (len(target_seg) + 1):  # +1 for last sentence
            print('Pick another article')
            print(f'len(targets) + 1 = {len(target_seg) + 1}')
            print(f'total_num_sentences - bad_sentences = {total_num_sentences - bad_sentences}')
        else:
            print('Finished comparison')
            print(f'Result at {result_file_path}')
            print(f'F1: {preds_stats.get_f1():.4}.')
            print(f'Accuracy: {preds_stats.get_accuracy():.4}.')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g., fake word2vec)', action='store_true')
    parser.add_argument('--model', help='Model to run - will import and run', required=True)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--folder', help='Folder with files to test on', required=True)
    parser.add_argument('--output_folder', help='Folder for results', required=True)
    parser.add_argument('--wiki', help='If the dataset is from Wikipedia', action='store_true')
    parser.add_argument('--manifesto', help='If the dataset is from Manifesto', action='store_true')
    parser.add_argument('--choi', help='If the dataset is from Choi', action='store_true')

    main(parser.parse_args())