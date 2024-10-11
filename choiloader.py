import torch
from torch.utils.data import Dataset
import numpy as np
import random
from text_manipulation import split_sentences, word_model, extract_sentence_words
import utils
import math
from pathlib import Path  # Use pathlib, which is built-in with Python 3
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

logger = utils.setup_logger(__name__, 'train.log')

def get_choi_files(path):
    all_objects = Path(path).rglob('*.ref')  # Use rglob for recursive file search
    files = [str(p) for p in all_objects if p.is_file()]
    return files


def custom_pad(s, max_length):
    s_length = s.size()[0]
    v = utils.maybe_cuda(s.unsqueeze(0).unsqueeze(0))
    padded = F.pad(v, (0, 0, 0, max_length - s_length))  # (1, 1, max_length, 300)
    shape = padded.size()
    return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

def pack_tensor(batch):
        
    sentences_per_doc = []
    all_batch_sentences = []
    for document in batch:
        all_batch_sentences.extend(document)
        sentences_per_doc.append(len(document))

    lengths = [s.size()[0] for s in all_batch_sentences]
    sort_order = np.argsort(lengths)[::-1]
    sorted_sentences = [all_batch_sentences[i] for i in sort_order]
    sorted_lengths = [s.size()[0] for s in sorted_sentences]

    max_length = max(lengths)
    logger.debug('Num sentences: %s, max sentence length: %s', 
                    sum(sentences_per_doc), max_length)

    padded_sentences = [custom_pad(s, max_length) for s in sorted_sentences]
    big_tensor = torch.cat(padded_sentences, 1)  # (max_length, batch size, 300)
    packed_tensor = pack_padded_sequence(big_tensor, sorted_lengths, enforce_sorted=False)
    return packed_tensor,sentences_per_doc,sort_order


def collate_fn(batch):
    batched_data = []
    batched_targets = []
    paths = []

    window_size = 1
    before_sentence_count = int(math.ceil((window_size - 1) / 2.0))  # Python 3 division
    after_sentence_count = window_size - before_sentence_count - 1

    for data, targets, path in batch:
        try:
            max_index = len(data)
            tensored_data = []
            for curr_sentence_index in range(0, len(data)):
                from_index = max(0, curr_sentence_index - before_sentence_count)
                to_index = min(curr_sentence_index + after_sentence_count + 1, max_index)
                sentences_window = [word for sentence in data[from_index:to_index] for word in sentence]
                tensored_data.append(torch.FloatTensor(np.concatenate(sentences_window)))
            tensored_targets = torch.zeros(len(data)).long()
            tensored_targets[torch.LongTensor(targets)] = 1
            tensored_targets = tensored_targets[:-1]
            batched_data.append(tensored_data)
            batched_targets.append(tensored_targets)
            paths.append(path)
        except Exception as e:
            logger.info(f'Exception "{e}" in file: "{path}"')
            logger.debug('Exception!', exc_info=True)
            continue

    packed_data,sentences_per_doc,sort_order = pack_tensor(batched_data)

    data = (packed_data,sentences_per_doc,sort_order,len(batch))
    
    return (data,batched_targets,paths)

def clean_paragraph(paragraph):
    cleaned_paragraph = paragraph.replace("'' ", " ").replace(" 's", "'s").replace("``", "").strip('\n')
    return cleaned_paragraph

def read_choi_file(path, word2vec, train, return_w2v_tensors=True, manifesto=False):
    separator = '========' if manifesto else '=========='
    with open(path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    paragraphs = [clean_paragraph(p) for p in raw_text.strip().split(separator) if len(p) > 5 and p != "\n"]
    if train:
        random.shuffle(paragraphs)

    targets = []
    new_text = []
    last_paragraph_sentence_idx = 0

    for paragraph in paragraphs:
        sentences = split_sentences(paragraph, 0) if manifesto else [s for s in paragraph.split('\n') if s.split()]
        if sentences:
            sentence_count = 0
            for sentence in sentences:
                words = extract_sentence_words(sentence)
                if len(words) == 0:
                    continue
                sentence_count += 1
                if return_w2v_tensors:
                    new_text.append([word_model(w, word2vec) for w in words])
                else:
                    new_text.append(words)

            last_paragraph_sentence_idx += sentence_count
            targets.append(last_paragraph_sentence_idx - 1)

    return new_text, targets, path

class ChoiDataset(Dataset):
    def __init__(self, root, word2vec, train=False, folder=False, manifesto=False, folders_paths=None):
        self.manifesto = manifesto
        if folders_paths is not None:
            self.textfiles = []
            for f in folders_paths:
                self.textfiles.extend(list(f.glob('*.ref')))
        elif folder:
            self.textfiles = get_choi_files(root)
        else:
            self.textfiles = list(Path(root).rglob('*.ref'))

        if len(self.textfiles) == 0:
            raise RuntimeError(f'Found 0 files in subfolders of: {root}')
        self.train = train
        self.root = root
        self.word2vec = word2vec

    def __getitem__(self, index):
        path = self.textfiles[index]
        return read_choi_file(path, self.word2vec, self.train, manifesto=self.manifesto)

    def __len__(self):
        return len(self.textfiles)