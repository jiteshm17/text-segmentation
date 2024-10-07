import unittest
from utils import unsort
import accuracy
import numpy as np
import text_manipulation

class LoaderTests(unittest.TestCase):
    def test_really_trivial(self):
        self.assertEqual(1 + 1, 2)

class PkTests(unittest.TestCase):
    def test_get_boundaries(self):
        sentences_class = [
            ("first sen.", 1),
            ("sec sen.", 1),
            ("third sen.", 0),
            ("forth sen.", 1),
            ("fifth sen.", 0),
            ("sixth sen.", 0),
            ("seventh sen.", 1)
        ]
        expected = [2, 2, 4, 6]
        result = accuracy.get_seg_boundaries(sentences_class)
        self.assertEqual(result, expected)

    def test_get_boundaries2(self):
        sentences_class = [
            ("first sen is 5 words.", 0),
            ("sec sen.", 0),
            ("third sen is a very very very long sentence.", 1),
            ("the forth one is a single segment.", 1)
        ]
        expected = [16, 6]
        result = accuracy.get_seg_boundaries(sentences_class)
        self.assertEqual(result, expected)

    def test_pk_perfect_seg(self):
        sentences_class = [
            ("first sen is 5 words.", 0),
            ("sec sen.", 0),
            ("third sen is a very very very long sentence.", 1),
            ("the forth one is a single segment.", 1)
        ]
        gold = accuracy.get_seg_boundaries(sentences_class)
        h = accuracy.get_seg_boundaries(sentences_class)

        for window_size in range(1, 15):
            acc = accuracy.pk(gold, h, window_size=window_size)
            self.assertEqual(acc, 1)

        acc = accuracy.pk(gold, h)
        self.assertEqual(acc, 1)

    def test_pk_false_neg(self):
        h = [("5 words sentence of data.", 0), ("2 sentences same seg.", 1)]
        gold = [("5 words sentence of data.", 1), ("2 sentences same seg.", 1)]

        gold = accuracy.get_seg_boundaries(gold)
        h = accuracy.get_seg_boundaries(h)

        window_size = 3
        comparison_count = 6

        acc = accuracy.pk(gold, h)
        self.assertEqual(acc, window_size / comparison_count)

        window_size = 4
        acc = accuracy.pk(gold, h)
        self.assertEqual(acc, window_size / comparison_count)

    def test_windiff(self):
        h = [
            ("5 words sentence of data.", 0),
            ("short.", 1),
            ("extra segmented sen.", 1),
            ("last and very very very very very long sen.", 1)
        ]

        gold = [
            ("5 words sentence of data.", 1),
            ("short.", 1),
            ("extra segmented sen.", 0),
            ("last and very very very very very long sen.", 1)
        ]

        gold = accuracy.get_seg_boundaries(gold)
        h = accuracy.get_seg_boundaries(h)

        window_size = 3
        acc = accuracy.win_diff(gold, h, window_size=window_size)
        self.assertEqual(float(acc), 0.6)

        window_size = 5
        expected = 1 - 8 / 13

        acc = accuracy.win_diff(gold, h, window_size=window_size)
        self.assertAlmostEqual(float(acc), expected, places=5)

class UnsortTests(unittest.TestCase):
    def test_unsort(self):
        x = np.random.randint(0, 100, 10)
        sort_order = np.argsort(x)
        unsort_order = unsort(sort_order)
        np.testing.assert_array_equal(x[sort_order][unsort_order], x)

class SentenceTokenizerTests(unittest.TestCase):
    def test_split_sentences(self):
        text = u"Hello, Mr. Trump, how do you do? What? Where? I don't i.e e.g Russia."
        expected = [
            u'Hello, Mr. Trump, how do you do?',
            u'What?',
            u'Where?',
            u"I don't i.e e.g Russia."
        ]
        result = text_manipulation.split_sentences(text)
        self.assertEqual(result, expected)

    def test_linebreaks(self):
        text = u'''Line one. Still line one.

        Line two. Can I span
        two lines?'''
        expected = [
            u'Line one.',
            u'Still line one.',
            u'Line two.',
            u'Can I span\n        two lines?'
        ]
        result = text_manipulation.split_sentences(text)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()