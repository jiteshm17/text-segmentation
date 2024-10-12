import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import maybe_cuda, setup_logger, unsort
import numpy as np
from times_profiler import profiler

logger = setup_logger(__name__, 'train.log')
profilerLogger = setup_logger("profilerLogger", 'profiler.log', True)


class Model(nn.Module):
    def __init__(self, input_size, hidden=128, num_layers=2):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden = hidden
        self.num_layers = num_layers
        

        self.sentence_encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden,
            num_layers=self.num_layers,
            dropout=0,
            bidirectional=True
        )

        self.sentence_lstm = nn.LSTM(
            input_size=self.hidden * 2,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        self.h2s = nn.Linear(hidden * 2, 2)
        self.criterion = nn.CrossEntropyLoss()


    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0, 0, max_document_length - d_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)
    
    
    def forward_sentence_encoding(self, x):
        # num_sequences = x.batch_sizes[0]
        packed_output, _ = self.sentence_encoder(x)
        padded_output, lengths = pad_packed_sequence(packed_output)  # (max sentence len, batch, 256)

        # maxes = maybe_cuda(torch.zeros(num_sequences, padded_output.size(2)))
        # for i in range(num_sequences):
        #     maxes[i, :] = torch.max(padded_output[:lengths[i], i, :], 0)[0]

        # Create a mask based on lengths
        mask = torch.arange(padded_output.size(0)).unsqueeze(1) < lengths.unsqueeze(0)
        mask = maybe_cuda(mask)
        
        # Mask padded values by setting them to a very negative value (so they don't affect the max computation)
        padded_output = padded_output.masked_fill(~mask.unsqueeze(2), float('-inf'))

        # Apply max pooling over the first dimension (time dimension) for each batch
        maxes, _ = torch.max(padded_output, dim=0)

        return maxes
    
    def forward(self, data):
        packed_tensor, sentences_per_doc, sort_order = data
        packed_tensor = maybe_cuda(packed_tensor)
        encoded_sentences = self.forward_sentence_encoding(packed_tensor)
        unsort_order = maybe_cuda(torch.LongTensor(unsort(sort_order)))
        unsorted_encodings = encoded_sentences.index_select(0, unsort_order)

        index = 0
        encoded_documents = []
        for sentences_count in sentences_per_doc:
            end_index = index + sentences_count
            encoded_documents.append(unsorted_encodings[index: end_index, :])
            index = end_index

        doc_sizes = [doc.size()[0] for doc in encoded_documents]
        max_doc_size = np.max(doc_sizes)
        ordered_document_idx = np.argsort(doc_sizes)[::-1]
        ordered_doc_sizes = sorted(doc_sizes)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [self.pad_document(d, max_doc_size) for d in ordered_documents]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes, enforce_sorted=False)
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs)
        padded_x, _ = pad_packed_sequence(sentence_lstm_output)  # (max sentence len, batch, 256)

        doc_outputs = []
        for i, doc_len in enumerate(ordered_doc_sizes):
            doc_outputs.append(padded_x[0:doc_len - 1, i, :])  # -1 to remove last prediction

        unsorted_doc_outputs = [doc_outputs[i] for i in unsort(ordered_document_idx)]
        sentence_outputs = torch.cat(unsorted_doc_outputs, 0)

        x = self.h2s(sentence_outputs)
        return x