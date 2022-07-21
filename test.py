import random
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle

from tqdm import tqdm
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore')

from modules import LSTMNet

CFG = {
    'NUM_WORKERS':0,
    'ANTIGEN_WINDOW':64,
    'ANTIGEN_MAX_LEN':64, # ANTIGEN_WINDOW와 ANTIGEN_MAX_LEN은 같아야합니다.
    'EPITOPE_MAX_LEN':80,
    'EPOCHS':100000,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':1024,
    'THRESHOLD':0.5,
    'SEED':41,
    'PATIENCE':20
}

alpha_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
    'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17,
    'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25, '<PAD>': 26,
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

embed = np.load('embed_matrix_1900.npy')
embed = np.concatenate((embed, np.expand_dims(np.zeros(embed.shape[1]),axis=0)), axis=0)

def get_custom_preprocessing(data_type, new_df):
    epitope_list = []
    left_antigen_list = []
    right_antigen_list = []
    total_antigen_list = []
    count = 0
    for epitope, antigen, s_p, e_p in tqdm(
            zip(new_df['epitope_seq'], new_df['antigen_seq'], new_df['start_position'], new_df['end_position'])):
        # epitope_pad = [26 for _ in range(CFG['EPITOPE_MAX_LEN'])]
        # left_antigen_pad = [26 for _ in range(CFG['ANTIGEN_MAX_LEN'])]
        # right_antigen_pad = [26 for _ in range(CFG['ANTIGEN_MAX_LEN'])]
        # epitope_pad = [26 for _ in range(CFG['EPITOPE_MAX_LEN'])]
        epitope_pad = [26 for _ in range(2*CFG['ANTIGEN_MAX_LEN']+CFG['EPITOPE_MAX_LEN'])]
        left_antigen_pad = [26 for _ in range(2*CFG['ANTIGEN_MAX_LEN']+CFG['EPITOPE_MAX_LEN'])]
        right_antigen_pad = [26 for _ in range(2*CFG['ANTIGEN_MAX_LEN']+CFG['EPITOPE_MAX_LEN'])]
        total_antigen_pad = [26 for _ in range(2*CFG['ANTIGEN_MAX_LEN']+CFG['EPITOPE_MAX_LEN'])]

        epitope = [alpha_map[x] for x in epitope]

        # Left antigen : [start_position-WINDOW : start_position]
        # Right antigen : [end_position : end_position+WINDOW]

        start_position = s_p - CFG['ANTIGEN_WINDOW'] - 1
        end_position = e_p + CFG['ANTIGEN_WINDOW']
        if start_position < 0:
            start_position = 0
        if end_position > len(antigen):
            end_position = len(antigen)

        # left / right antigen sequence 추출
        left_antigen = antigen[int(start_position): int(s_p) - 1]
        left_antigen = [alpha_map[x] for x in left_antigen]

        right_antigen = antigen[int(e_p): int(end_position)]
        right_antigen = [alpha_map[x] for x in right_antigen]

        if CFG['EPITOPE_MAX_LEN'] < len(epitope):
            epitope_pad[:len(epitope)] = epitope[:CFG['EPITOPE_MAX_LEN']]
        else:
            epitope_pad[:len(epitope)] = epitope[:]

        left_antigen_pad[:len(left_antigen)] = left_antigen[:]
        right_antigen_pad[:len(right_antigen)] = right_antigen[:]
        count+=1
        if count >= 93:
            total_antigen_pad[:(len(left_antigen)+len(epitope)+len(right_antigen))] = left_antigen + epitope + right_antigen
        # total_antigen_pad[:len(left_antigen)] = left_antigen[:]
        # total_antigen_pad[len(left_antigen):(len(left_antigen)+len(epitope))] = epitope[:]
        # total_antigen_pad[(len(left_antigen)+len(epitope)):(len(left_antigen)+len(epitope)+len(right_antigen))] = right_antigen[:]

        epitope_list.append(epitope_pad)
        left_antigen_list.append(left_antigen_pad)
        right_antigen_list.append(right_antigen_pad)
        total_antigen_list.append(total_antigen_pad)

    label_list = None
    if data_type != 'test':
        label_list = []
        for label in new_df['label']:
            label_list.append(label)
    print(f'{data_type} dataframe preprocessing was done.')
    return epitope_list, left_antigen_list, right_antigen_list, total_antigen_list, label_list

class BaseModel(nn.Module):
    def __init__(self,
                 epitope_length=CFG['EPITOPE_MAX_LEN'],
                 epitope_emb_node=10,
                 epitope_hidden_dim=1900,
                 total_hidden_dim=1900,
                 left_antigen_length=CFG['ANTIGEN_MAX_LEN'],
                 left_antigen_emb_node=10,
                 left_antigen_hidden_dim=1900,
                 right_antigen_length=CFG['ANTIGEN_MAX_LEN'],
                 right_antigen_emb_node=10,
                 right_antigen_hidden_dim=1900,
                 lstm_bidirect=False
                 ):
        super(BaseModel, self).__init__()
        self.lstm_bidirect = lstm_bidirect
        # Embedding Layer
        self.total_embed = nn.Embedding(num_embeddings=27,
                                        embedding_dim=epitope_emb_node,
                                        padding_idx=26
                                        ).from_pretrained(torch.Tensor(embed))
        '''
        self.epitope_embed = nn.Embedding(num_embeddings=27,
                                          embedding_dim=epitope_emb_node,
                                          padding_idx=26
                                          )
        self.left_antigen_embed = nn.Embedding(num_embeddings=27,
                                               embedding_dim=left_antigen_emb_node,
                                               padding_idx=26
                                               )
        self.right_antigen_embed = nn.Embedding(num_embeddings=27,
                                                embedding_dim=right_antigen_emb_node,
                                                padding_idx=26
                                                )
        '''

        # LSTM
        self.forlstm = LSTMNet(model_path='1900_weights')
        self.forlstm_left = LSTMNet(model_path='1900_weights')
        self.forlstm_right = LSTMNet(model_path='1900_weights')
        # self.backlstm = LSTMNet(model_path='1900_weights')
        '''
        self.epitope_lstm = nn.LSTM(input_size=epitope_emb_node,
                                    hidden_size=epitope_hidden_dim,
                                    batch_first=True,
                                    bidirectional=lstm_bidirect
                                    )
        self.left_antigen_lstm = nn.LSTM(input_size=left_antigen_emb_node,
                                         hidden_size=left_antigen_hidden_dim,
                                         batch_first=True,
                                         bidirectional=lstm_bidirect
                                         )
        self.right_antigen_lstm = nn.LSTM(input_size=right_antigen_emb_node,
                                          hidden_size=right_antigen_hidden_dim,
                                          batch_first=True,
                                          bidirectional=lstm_bidirect
                                          )
        '''
        # Classifier
        if lstm_bidirect:
            # in_channels = 2 * (epitope_hidden_dim + left_antigen_hidden_dim + right_antigen_hidden_dim)
            in_channels = 2 * (epitope_hidden_dim) + total_hidden_dim
        else:
            in_channels = epitope_hidden_dim + left_antigen_hidden_dim + right_antigen_hidden_dim
            # in_channels = epitope_hidden_dim + total_hidden_dim

        self.classifier = nn.Sequential(
            nn.LeakyReLU(True),
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, in_channels // 10),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(in_channels // 10),
            nn.Linear(in_channels // 10, 1)
        )

    def forward(self, epitope_x, left_antigen_x, right_antigen_x, total_antigen_x):
        # BATCH_SIZE = epitope_x.size(0)
        # Get Lengths and Embedding Vector
        epitope_lengths = torch.sum(epitope_x != alpha_map['<PAD>'], axis=1).type(torch.IntTensor)
        # epitope is always >0
        left_lengths = torch.sum(left_antigen_x != alpha_map['<PAD>'], axis=1).type(torch.IntTensor)
        left_lengths += torch.where(left_lengths == 0, 1, 0)
        right_lengths = torch.sum(right_antigen_x != alpha_map['<PAD>'], axis=1).type(torch.IntTensor)
        right_lengths += torch.where(right_lengths == 0, 1, 0)
        total_lengths = left_lengths + epitope_lengths + right_lengths
        '''
        if self.lstm_bidirect:
            flipped_epitope_x = torch.stack(
                [torch.cat((torch.flip(epitope_x[t,:epitope_lengths[t]],dims=[0]), epitope_x[t, epitope_lengths[t]:])) for t in range(epitope_x.shape[0])]
            )
            flipped_epitope_x = self.total_embed(flipped_epitope_x)
        '''

        epitope_x = self.total_embed(epitope_x)
        # total_antigen_x = self.total_embed(total_antigen_x)
        left_antigen_x = self.total_embed(left_antigen_x)
        right_antigen_x = self.total_embed(right_antigen_x)

        # LSTM
        # epitope_x = torch.nn.utils.rnn.pack_padded_sequence(epitope_x, epitope_lengths, batch_first=True,
        #                                                     enforce_sorted=False)
        epitope_hidden_forward, _ = self.forlstm(epitope_x)
        epitope_hidden_forward = epitope_hidden_forward[torch.arange(epitope_hidden_forward.shape[0]),
                                 (epitope_lengths - 1).type(torch.LongTensor), :]

        left_antigen_hidden_forward, _ = self.forlstm_left(left_antigen_x)
        left_antigen_hidden_forward = left_antigen_hidden_forward[torch.arange(left_antigen_hidden_forward.shape[0]),
                                      (left_lengths - 1).type(torch.LongTensor), :]

        right_antigen_hidden_forward, _ = self.forlstm_right(right_antigen_x)
        right_antigen_hidden_forward = right_antigen_hidden_forward[torch.arange(right_antigen_hidden_forward.shape[0]),
                                       (right_lengths - 1).type(torch.LongTensor), :]

        '''
        total_hidden_forward, _ = self.forlstm(total_antigen_x)
        total_hidden_forward = total_hidden_forward[torch.arange(total_hidden_forward.shape[0]),
                                 (total_lengths - 1).type(torch.LongTensor), :]
        '''
        '''
        if self.lstm_bidirect:
            epitope_hidden_backward, _ = self.backlstm(flipped_epitope_x)
            epitope_hidden_backward = epitope_hidden_backward[torch.arange(epitope_hidden_backward.shape[0]),
                                     (epitope_lengths - 1).type(torch.LongTensor), :]

            epitope_hidden = torch.cat([epitope_hidden_forward, epitope_hidden_backward], axis=-1)

        else:
            epitope_hidden = epitope_hidden_forward
        '''

        epitope_hidden = epitope_hidden_forward
        epitope_hidden = torch.cat([left_antigen_hidden_forward, epitope_hidden, right_antigen_hidden_forward], dim=-1)

        # epitope_hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(epitope_hidden, batch_first=True)
        # epitope_hidden = epitope_hidden[torch.arange(epitope_hidden.shape[0]),
        #                  (epitope_lengths - 1).type(torch.LongTensor), :]
        '''
        epitope_x = torch.nn.utils.rnn.pack_padded_sequence(epitope_x, epitope_lengths, batch_first=True, enforce_sorted=False)
        epitope_hidden, _ = self.epitope_lstm(epitope_x)
        epitope_hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(epitope_hidden, batch_first=True)
        epitope_hidden = epitope_hidden[torch.arange(epitope_hidden.shape[0]), (epitope_lengths-1).type(torch.LongTensor), :]

        left_antigen_x = torch.nn.utils.rnn.pack_padded_sequence(left_antigen_x, left_lengths, batch_first=True, enforce_sorted=False)
        left_antigen_hidden, _ = self.left_antigen_lstm(left_antigen_x)
        left_antigen_hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(left_antigen_hidden, batch_first=True)
        left_antigen_hidden = left_antigen_hidden[torch.arange(left_antigen_hidden.shape[0]), (left_lengths-1).type(torch.LongTensor), :]

        right_antigen_x = torch.nn.utils.rnn.pack_padded_sequence(right_antigen_x, right_lengths, batch_first=True, enforce_sorted=False)
        right_antigen_hidden, _ = self.right_antigen_lstm(right_antigen_x)
        right_antigen_hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(right_antigen_hidden, batch_first=True)
        right_antigen_hidden = right_antigen_hidden[torch.arange(right_antigen_hidden.shape[0]), (right_lengths-1).type(torch.LongTensor), :]
        '''

        # Feature Concat -> Binary Classifier
        x = epitope_hidden
        # x = torch.cat([epitope_hidden, left_antigen_hidden, right_antigen_hidden], axis=-1)
        x = self.classifier(x).view(-1)
        return x

class CustomDataset(Dataset):
    def __init__(self, epitope_list, left_antigen_list, right_antigen_list, total_antigen_list, label_list):
        self.epitope_list = epitope_list
        self.left_antigen_list = left_antigen_list
        self.right_antigen_list = right_antigen_list
        self.total_antigen_list = total_antigen_list
        self.label_list = label_list

    def __getitem__(self, index):
        self.epitope = self.epitope_list[index]
        self.left_antigen = self.left_antigen_list[index]
        self.right_antigen = self.right_antigen_list[index]
        self.total_antigen = self.total_antigen_list[index]

        if self.label_list is not None:
            self.label = self.label_list[index]
            return torch.tensor(self.epitope), torch.tensor(self.left_antigen), torch.tensor(
                self.right_antigen), torch.tensor(self.total_antigen), self.label
        else:
            return torch.tensor(self.epitope), torch.tensor(self.left_antigen), torch.tensor(self.right_antigen), torch.tensor(self.total_antigen)

    def __len__(self):
        return len(self.epitope_list)

test_df = pd.read_csv('./test.csv')
# test_df = test_df[test_df['end_position'] - test_df['start_position'] + 1 <= 25]
test_epitope_list, test_left_antigen_list, test_right_antigen_list, total_antigen_list, test_label_list = get_custom_preprocessing('test', test_df)

test_dataset = CustomDataset(test_epitope_list, test_left_antigen_list, test_right_antigen_list, total_antigen_list, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'])

model = BaseModel()
best_checkpoint = torch.load('./best_model_base.pth')
model.load_state_dict(best_checkpoint)
model.eval()
model.to(device)


def inference(model, test_loader, device):
    model.eval()
    pred_proba_label = []
    with torch.no_grad():
        for epitope_seq, left_antigen_seq, right_antigen_seq, total_antigen_seq in tqdm(iter(test_loader)):
            epitope_seq = epitope_seq.to(device)
            left_antigen_seq = left_antigen_seq.to(device)
            right_antigen_seq = right_antigen_seq.to(device)
            total_antigen_seq = total_antigen_seq.to(device)

            model_pred = model(epitope_seq, left_antigen_seq, right_antigen_seq, total_antigen_seq)
            model_pred = torch.sigmoid(model_pred).to('cpu')

            pred_proba_label += model_pred.tolist()
        torch.cuda.empty_cache()
    pred_label = np.where(np.array(pred_proba_label) > CFG['THRESHOLD'], 1, 0)
    return pred_label

preds = inference(model, test_loader, device)

# submission
submit = pd.read_csv('./sample_submission.csv')
submit['label'] = preds

submit.to_csv('./submit.csv', index=False)
print('Done.')