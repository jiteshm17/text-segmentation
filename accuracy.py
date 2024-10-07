import segeval as seg
import numpy as np

def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums

class Accuracy:
    def __init__(self, threshold=0.3):
        self.pk_to_weight = []
        self.windiff_to_weight = []
        self.threshold = threshold

    def update(self, h, gold, sentences_length=None):
        h_boundaries = self.get_seg_boundaries(h, sentences_length)
        gold_boundaries = self.get_seg_boundaries(gold, sentences_length)
        pk, count_pk = self.pk(h_boundaries, gold_boundaries)
        windiff, count_wd = -1, 400  # Placeholder for windiff calculation

        if pk != -1:
            self.pk_to_weight.append((pk, count_pk))
        else:
            print('pk error')

        if windiff != -1:
            self.windiff_to_weight.append((windiff, count_wd))

    def get_seg_boundaries(self, classifications, sentences_length=None):
        """
        :param classifications: list of tuples, each tuple is a sentence and its class (1 if the sentence starts a segment, 0 otherwise).
        :param sentences_length: list of sentence lengths (optional)
        :return: boundaries of segmentation for pk method.
        """
        curr_seg_length = 0
        boundaries = []
        for i, classification in enumerate(classifications):
            is_split_point = bool(classifications[i])
            add_to_current_segment = 1 if sentences_length is None else sentences_length[i]
            curr_seg_length += add_to_current_segment
            if is_split_point:
                boundaries.append(curr_seg_length)
                curr_seg_length = 0

        return boundaries

    def pk(self, h, gold, window_size=-1):
        """
        :param h: hypothesis segmentation
        :param gold: gold segmentation
        :param window_size: optional window size
        :return: pk accuracy
        """
        if window_size != -1:
            false_seg_count, total_count = seg.pk(h, gold, window_size=window_size, return_parts=True)
        else:
            false_seg_count, total_count = seg.pk(h, gold, return_parts=True)

        if total_count == 0:
            false_prob = -1
        else:
            false_prob = float(false_seg_count) / total_count

        return false_prob, total_count

    def win_diff(self, h, gold, window_size=-1):
        """
        :param h: hypothesis segmentation
        :param gold: gold segmentation
        :param window_size: optional window size
        :return: win_diff accuracy
        """
        if window_size != -1:
            false_seg_count, total_count = seg.window_diff(h, gold, window_size=window_size, return_parts=True)
        else:
            false_seg_count, total_count = seg.window_diff(h, gold, return_parts=True)

        if total_count == 0:
            false_prob = -1
        else:
            false_prob = float(false_seg_count) / total_count

        return false_prob, total_count

    def calc_accuracy(self):
        pk = sum(pw[0] * pw[1] for pw in self.pk_to_weight) / sum(pw[1] for pw in self.pk_to_weight) if self.pk_to_weight else -1.0
        windiff = sum(pw[0] * pw[1] for pw in self.windiff_to_weight) / sum(pw[1] for pw in self.windiff_to_weight) if self.windiff_to_weight else -1.0

        return pk, windiff