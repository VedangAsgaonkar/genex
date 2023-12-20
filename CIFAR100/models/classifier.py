import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence as pad
import torch.nn.functional as F


# class Classifier(nn.Module):
#     def __init__(self, hidden_size, num_classes):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 32),
#             nn.ReLU(),
#             nn.Linear(32, num_classes)
#         )
#         # self.fc = nn.Sequential(
#         #     nn.Linear(hidden_size, num_classes)
#         # )
#         self.logits = nn.Softmax(dim=1)

#     def forward(self, x):
#         return self.logits(self.fc(x))


# class Classifier(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size, bias=True,
#             dropout=False, p=0, group_norm=0, batch_norm=False):
#         super(Classifier, self).__init__()
#         self.layers = [nn.Flatten()]
#         self.n_features = int(input_size / 2)
#         in_size = input_size
#         cnt = 0
#         for hidden_size in hidden_sizes:
#             # print(in_size, hidden_size, bias,": args")
#             self.layers.append(nn.Linear(in_size, hidden_size, bias=bias))
#             if group_norm > 0 and cnt == 0:
#                 cnt += 1
#                 self.w0 = self.layers[-1].weight
#                 print(self.w0.size())
#                 assert self.w0.size()[1] == input_size
#             if batch_norm:
#                 print("Batchnorm")
#                 self.layers.append(nn.BatchNorm1d(hidden_size))
#             self.layers.append(nn.ReLU())
#             if dropout: # for classifier
#                 print("Dropout!")
#                 assert p > 0 and p < 1
#                 self.layers.append(nn.Dropout(p=p))
#             in_size = hidden_size
#         self.layers.append(nn.Linear(in_size, output_size, bias=bias))
#         if batch_norm: # FIXME is it good?
#             print("Batchnorm")
#             self.layers.append(nn.BatchNorm1d(output_size))
#         self.layers = nn.ModuleList(self.layers)

#         self.output_size = output_size

#         self.logits = nn.Softmax(dim = 1)


    # def forward(self, x, length=None):
    #     # print(x.shape)
    #     # print(self.layers)
    #     for layer in self.layers:
    #         x = layer(x)
    #     # return x
        # return self.logits(x)



class Classifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True,
            dropout=False, p=0, group_norm=0, batch_norm=False):
        super(Classifier, self).__init__()
        self.layers = [nn.Flatten()]
        # self.n_features = int(input_size / 2)
        in_size = input_size
        cnt = 0
        for hidden_size in hidden_sizes:
            # print(in_size, hidden_size, bias,": args")
            # print(in_size, hidden_size, bias)
            self.layers.append(nn.Linear(in_size, hidden_size, bias=bias))
            if group_norm > 0 and cnt == 0:
                cnt += 1
                self.w0 = self.layers[-1].weight
                print(self.w0.size())
                assert self.w0.size()[1] == input_size
            if batch_norm:
                print("Batchnorm")
                # self.layers.append(nn.BatchNorm1d(hidden_size, affine=False))
            self.layers.append(nn.ReLU())
            if dropout: # for classifier
                print("Dropout!")
                assert p > 0 and p < 1
                self.layers.append(nn.Dropout(p=p))
            in_size = hidden_size
        self.layers.append(nn.Linear(in_size, output_size, bias=bias))
        if batch_norm: # FIXME is it good?
            print("Batchnorm")
            self.layers.append(nn.BatchNorm1d(output_size, affine = False))
        self.layers = nn.ModuleList(self.layers)

        self.output_size = output_size

        self.logits = nn.Softmax(dim = 1)


    def forward(self, x, length=None):
        # print(x.shape)
        # print(self.layers)
        for layer in self.layers:
            x = layer(x)
        return x
        # return self.logits(x)

class ChoiceClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True,
            dropout=False, p=0, group_norm=0, batch_norm=False):
        super(ChoiceClassifier, self).__init__()
        self.layers = [nn.Flatten()]
        # self.n_features = int(input_size / 2)
        in_size = input_size
        cnt = 0
        for hidden_size in hidden_sizes:
            # print(in_size, hidden_size, bias,": args")
            # print(in_size, hidden_size, bias)
            self.layers.append(nn.Linear(in_size, hidden_size, bias=bias))
            if group_norm > 0 and cnt == 0:
                cnt += 1
                self.w0 = self.layers[-1].weight
                print(self.w0.size())
                assert self.w0.size()[1] == input_size
            if batch_norm:
                print("Batchnorm")
                # self.layers.append(nn.BatchNorm1d(hidden_size, affine=False))
            self.layers.append(nn.ReLU())
            if dropout: # for classifier
                print("Dropout!")
                assert p > 0 and p < 1
                self.layers.append(nn.Dropout(p=p))
            in_size = hidden_size
        self.layers.append(nn.Linear(in_size, output_size, bias=bias))
        if batch_norm: # FIXME is it good?
            print("Batchnorm")
            self.layers.append(nn.BatchNorm1d(output_size, affine = False))
        self.layers = nn.ModuleList(self.layers)

        self.output_size = output_size

        self.logits = nn.Softmax(dim = 1)


    def forward(self, x, length=None):
        # print(x.shape)
        # print(self.layers)
        for layer in self.layers:
            x = layer(x)
        return x
        # return self.logits(x)
