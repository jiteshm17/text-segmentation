import json
import logging
import sys
import numpy as np
import random
from pathlib import Path  # Updated to use pathlib (pathlib2 is not needed in Python 3)
from shutil import copy

config = {}

def read_config_file(path='config.json'):
    global config
    with open(path, 'r') as f:
        config.update(json.load(f))

def maybe_cuda(x, is_cuda=None):
    global config
    if is_cuda is None and 'cuda' in config:
        is_cuda = config['cuda']
    if is_cuda:
        return x.cuda()
    return x

def setup_logger(logger_name, filename, delete_old=False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(filename, mode='w' if delete_old else 'a')
    file_handler.setLevel(logging.DEBUG)
    stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger

def unsort(sort_order):
    result = [-1] * len(sort_order)
    for i, index in enumerate(sort_order):
        result[index] = i
    return result

class F1:
    def __init__(self, ner_size):
        self.ner_size = ner_size
        self.tp = np.zeros(ner_size + 1)
        self.fp = np.zeros(ner_size + 1)
        self.fn = np.zeros(ner_size + 1)

    def add(self, preds, targets, length):
        prediction = np.argmax(preds, axis=2)
        for i in range(len(targets)):
            for j in range(length[i]):
                if targets[i, j] == prediction[i, j]:
                    self.tp[targets[i, j]] += 1
                else:
                    self.fp[targets[i, j]] += 1
                    self.fn[prediction[i, j]] += 1

        unnamed_entity = self.ner_size - 1
        for i in range(self.ner_size):
            if i != unnamed_entity:
                self.tp[self.ner_size] += self.tp[i]
                self.fp[self.ner_size] += self.fp[i]
                self.fn[self.ner_size] += self.fn[i]

    def score(self):
        precision = np.divide(self.tp, self.tp + self.fp, out=np.zeros_like(self.tp), where=self.tp + self.fp != 0)
        recall = np.divide(self.tp, self.tp + self.fn, out=np.zeros_like(self.tp), where=self.tp + self.fn != 0)
        fscore = 2 * precision * recall / (precision + recall + 1e-8)  # Avoid division by zero
        print(fscore)
        return fscore[self.ner_size]

class predictions_analysis:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, predictions, targets):
        self.tp += ((predictions == targets) & (predictions == 1)).sum()
        self.tn += ((predictions == targets) & (predictions == 0)).sum()
        self.fp += ((predictions != targets) & (predictions == 1)).sum()
        self.fn += ((predictions != targets) & (predictions == 0)).sum()

    def calc_recall(self):
        return np.divide(self.tp, self.tp + self.fn) if self.tp + self.fn != 0 else -1

    def calc_precision(self):
        return np.divide(self.tp, self.tp + self.fp) if self.tp + self.fp != 0 else -1

    def get_f1(self):
        if self.tp + self.fp == 0 or self.tp + self.fn == 0:
            return 0.0
        precision = self.calc_precision()
        recall = self.calc_recall()
        return 2 * precision * recall / (precision + recall + 1e-8) if precision + recall != 0 else 0.0

    def get_accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return np.divide(self.tp + self.tn, total) if total != 0 else 0.0

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0

def get_random_files(count, input_folder, output_folder, specific_section=True):
    files = Path(input_folder).glob('*/*/*/*') if specific_section else Path(input_folder).glob('*/*/*/*/*')
    file_paths = list(files)
    random_paths = random.sample(file_paths, count)
    for random_path in random_paths:
        output_path = Path(output_folder).joinpath(random_path.name)
        copy(random_path, output_path)