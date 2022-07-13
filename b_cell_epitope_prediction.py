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

torch.cuda.is_available()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

embed = np.load('embed_matrix_1900.npy')
embed = np.concatenate((embed, np.expand_dims(np.zeros(embed.shape[1]),axis=0)), axis=0)

CFG = {
    'NUM_WORKERS':0,
    'ANTIGEN_WINDOW':128,
    'ANTIGEN_MAX_LEN':128, # ANTIGEN_WINDOW와 ANTIGEN_MAX_LEN은 같아야합니다.
    'EPITOPE_MAX_LEN':256,
    'EPOCHS':100000,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':1024,
    'THRESHOLD':0.5,
    'SEED':41,
    'PATIENCE':20
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

alpha_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
    'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17,
    'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25, '<PAD>': 26,
}

def get_preprocessing(data_type, new_df):

    epitope_list = []
    left_antigen_list = []
    right_antigen_list = []

    for epitope, antigen, s_p, e_p in tqdm(
            zip(new_df['epitope_seq'], new_df['antigen_seq'], new_df['start_position'], new_df['end_position'])):
        epitope_pad = [26 for _ in range(CFG['EPITOPE_MAX_LEN'])]
        left_antigen_pad = [26 for _ in range(CFG['ANTIGEN_MAX_LEN'])]
        right_antigen_pad = [26 for _ in range(CFG['ANTIGEN_MAX_LEN'])]

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

        epitope_list.append(epitope_pad)
        left_antigen_list.append(left_antigen_pad)
        right_antigen_list.append(right_antigen_pad)

    label_list = None
    if data_type != 'test':
        label_list = []
        for label in new_df['label']:
            label_list.append(label)
    print(f'{data_type} dataframe preprocessing was done.')
    return epitope_list, left_antigen_list, right_antigen_list, label_list

all_df = pd.read_csv('./train.csv')
# Split Train : Validation = 0.8 : 0.2
# train_len = int(len(all_df)*0.8)
train_positive_len = int(len(all_df[all_df['label']==1])*0.7)
train_negative_len = int(len(all_df[all_df['label']==0])*0.7)
train_positive = all_df[all_df['label']==1]
train_negative = all_df[all_df['label']==0]
train_df = train_positive.iloc[:train_positive_len]
val_df = train_positive.iloc[train_positive_len:]
train_df = train_df.append(train_negative[:train_negative_len], ignore_index=True)
val_df = val_df.append(train_negative[train_negative_len:], ignore_index=True)

train_epitope_list, train_left_antigen_list, train_right_antigen_list, train_label_list = get_preprocessing('train', train_df)
val_epitope_list, val_left_antigen_list, val_right_antigen_list, val_label_list = get_preprocessing('val', val_df)


class CustomDataset(Dataset):
    def __init__(self, epitope_list, left_antigen_list, right_antigen_list, label_list):
        self.epitope_list = epitope_list
        self.left_antigen_list = left_antigen_list
        self.right_antigen_list = right_antigen_list
        self.label_list = label_list

    def __getitem__(self, index):
        self.epitope = self.epitope_list[index]
        self.left_antigen = self.left_antigen_list[index]
        self.right_antigen = self.right_antigen_list[index]

        if self.label_list is not None:
            self.label = self.label_list[index]
            return torch.tensor(self.epitope), torch.tensor(self.left_antigen), torch.tensor(
                self.right_antigen), self.label
        else:
            return torch.tensor(self.epitope), torch.tensor(self.left_antigen), torch.tensor(self.right_antigen)

    def __len__(self):
        return len(self.epitope_list)

train_dataset = CustomDataset(train_epitope_list, train_left_antigen_list, train_right_antigen_list, train_label_list)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'])

val_dataset = CustomDataset(val_epitope_list, val_left_antigen_list, val_right_antigen_list, val_label_list)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'])

# epitope_seq, left_antigen_seq, right_antigen_seq, label
a,b,c,d = train_dataset[0]
print(b)

# epitope_seq
print(a.shape, b.shape, c.shape)


class BaseModel(nn.Module):
    def __init__(self,
                 epitope_length=CFG['EPITOPE_MAX_LEN'],
                 epitope_emb_node=10,
                 epitope_hidden_dim=64,
                 left_antigen_length=CFG['ANTIGEN_MAX_LEN'],
                 left_antigen_emb_node=10,
                 left_antigen_hidden_dim=64,
                 right_antigen_length=CFG['ANTIGEN_MAX_LEN'],
                 right_antigen_emb_node=10,
                 right_antigen_hidden_dim=64,
                 lstm_bidirect=True
                 ):
        super(BaseModel, self).__init__()
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

        # Classifier
        if lstm_bidirect:
            in_channels = 2 * (epitope_hidden_dim + left_antigen_hidden_dim + right_antigen_hidden_dim)
        else:
            in_channels = epitope_hidden_dim + left_antigen_hidden_dim + right_antigen_hidden_dim

        self.classifier = nn.Sequential(
            nn.LeakyReLU(True),
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, in_channels // 4),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(in_channels // 4),
            nn.Linear(in_channels // 4, 1)
        )

    def forward(self, epitope_x, left_antigen_x, right_antigen_x):
        # BATCH_SIZE = epitope_x.size(0)
        # Get Lengths and Embedding Vector
        epitope_lengths = torch.sum(epitope_x != alpha_map['<PAD>'], axis=1).type(torch.IntTensor)
        # epitope is always >0
        left_lengths = torch.sum(left_antigen_x != alpha_map['<PAD>'], axis=1).type(torch.IntTensor)
        left_lengths += torch.where(left_lengths==0, 1, 0)
        right_lengths = torch.sum(right_antigen_x != alpha_map['<PAD>'], axis=1).type(torch.IntTensor)
        right_lengths += torch.where(right_lengths==0, 1, 0)

        epitope_x = self.total_embed(epitope_x)
        left_antigen_x = self.total_embed(left_antigen_x)
        right_antigen_x = self.total_embed(right_antigen_x)

        # LSTM
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

        # Feature Concat -> Binary Classifier
        x = torch.cat([epitope_hidden, left_antigen_hidden, right_antigen_hidden], axis=-1)
        x = self.classifier(x).view(-1)
        return x


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    counter = 0
    best_val_f1 = 0
    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for epitope_seq, left_antigen_seq, right_antigen_seq, label in tqdm(iter(train_loader)):
            epitope_seq = epitope_seq.to(device)
            left_antigen_seq = left_antigen_seq.to(device)
            right_antigen_seq = right_antigen_seq.to(device)
            label = label.float().to(device)

            optimizer.zero_grad()

            output = model(epitope_seq, left_antigen_seq, right_antigen_seq)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if scheduler is not None:
                scheduler.step()

        val_loss, val_f1 = validation(model, val_loader, criterion, device)
        print(
            f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val F1 : [{val_f1:.5f}]')

        if best_val_f1 < val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), './best_model.pth', _use_new_zipfile_serialization=False)
            print('Model Saved.')
            counter=0
        else:
            counter +=1
            print('model not improved... counter ', counter)
            if counter >= CFG['PATIENCE']:
                print('early stopping')
                break
    return best_val_f1


def validation(model, val_loader, criterion, device):
    model.eval()
    pred_proba_label = []
    true_label = []
    val_loss = []
    with torch.no_grad():
        for epitope_seq, left_antigen_seq, right_antigen_seq, label in tqdm(iter(val_loader)):
            epitope_seq = epitope_seq.to(device)
            left_antigen_seq = left_antigen_seq.to(device)
            right_antigen_seq = right_antigen_seq.to(device)
            label = label.float().to(device)

            model_pred = model(epitope_seq, left_antigen_seq, right_antigen_seq)
            loss = criterion(model_pred, label)
            model_pred = torch.sigmoid(model_pred).to('cpu')

            pred_proba_label += model_pred.tolist()
            true_label += label.to('cpu').tolist()

            val_loss.append(loss.item())

    pred_label = np.where(np.array(pred_proba_label) > CFG['THRESHOLD'], 1, 0)
    val_f1 = f1_score(true_label, pred_label, average='macro')
    return np.mean(val_loss), val_f1

model = BaseModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*CFG['EPOCHS'], eta_min=0)

best_score = train(model, optimizer, train_loader, val_loader, scheduler, device)
print(f'Best Validation F1 Score : [{best_score:.5f}]')

test_df = pd.read_csv('./test.csv')
test_epitope_list, test_left_antigen_list, test_right_antigen_list, test_label_list = get_preprocessing('test', test_df)

test_dataset = CustomDataset(test_epitope_list, test_left_antigen_list, test_right_antigen_list, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'])

model = BaseModel()
best_checkpoint = torch.load('./best_model.pth')
model.load_state_dict(best_checkpoint)
model.eval()
model.to(device)


def inference(model, test_loader, device):
    model.eval()
    pred_proba_label = []
    with torch.no_grad():
        for epitope_seq, left_antigen_seq, right_antigen_seq in tqdm(iter(test_loader)):
            epitope_seq = epitope_seq.to(device)
            left_antigen_seq = left_antigen_seq.to(device)
            right_antigen_seq = right_antigen_seq.to(device)

            model_pred = model(epitope_seq, left_antigen_seq, right_antigen_seq)
            model_pred = torch.sigmoid(model_pred).to('cpu')

            pred_proba_label += model_pred.tolist()

    pred_label = np.where(np.array(pred_proba_label) > CFG['THRESHOLD'], 1, 0)
    return pred_label

preds = inference(model, test_loader, device)

# submission
submit = pd.read_csv('./sample_submission.csv')
submit['label'] = preds

submit.to_csv('./submit.csv', index=False)
print('Done.')