import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
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

        # Create a mask based on lengths
        mask = torch.arange(padded_output.size(0)).unsqueeze(1) < lengths.unsqueeze(0)
        mask = maybe_cuda(mask)
        
        # Mask padded values by setting them to a very negative value (so they don't affect the max computation)
        padded_output = padded_output.masked_fill(~mask.unsqueeze(2), float('-inf'))

        # Apply max pooling over the first dimension (time dimension) for each batch
        maxes, _ = torch.max(padded_output, dim=0)

        return maxes
    

    def forward_helper(self, sentences_per_doc, unsorted_encodings):
        
        # Step 3: Efficiently split the unsorted_encodings into separate documents using tensor operations
        sentences_per_doc = maybe_cuda(torch.LongTensor(sentences_per_doc))
        encoded_documents = torch.split(unsorted_encodings, sentences_per_doc.tolist())

        # Step 4: Calculate maximum document size and pad documents in one go
        padded_docs = pad_sequence(encoded_documents, batch_first=True)

        # Step 5: Pack the padded documents for LSTM processing
        packed_docs = pack_padded_sequence(padded_docs, sentences_per_doc, batch_first=True, enforce_sorted=False)

        # Step 6: Pass through document-level LSTM
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs)

        # Step 7: Unpack the LSTM output
        padded_x, _ = pad_packed_sequence(sentence_lstm_output, batch_first=True)

        # Step 8: Select the final hidden states (excluding last prediction) without using a loop
        doc_outputs = [padded_x[i, :doc_len-1, :] for i, doc_len in enumerate(sentences_per_doc.tolist())]

        # Step 9: Concatenate the outputs into one tensor
        sentence_outputs = torch.cat(doc_outputs, dim=0)

        return sentence_outputs
    
    def forward(self, data):
        packed_tensor, sentences_per_doc, sort_order = data
        packed_tensor = maybe_cuda(packed_tensor)
        encoded_sentences = self.forward_sentence_encoding(packed_tensor)
        unsort_order = maybe_cuda(torch.LongTensor(unsort(sort_order)))
        unsorted_encodings = encoded_sentences.index_select(0, unsort_order)
        sentence_outputs = self.forward_helper(sentences_per_doc, unsorted_encodings)
        x = self.h2s(sentence_outputs)
        return x