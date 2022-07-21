import tensorflow as tf
from tensorflow.keras import Sequential, losses, optimizers, activations
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm

from tensorflow.keras.utils import Sequence
import pandas as pd
import sys
sys.path.append('../')
# from data_utils import aa_seq_to_int, int_to_aa, bucketbatchpad
import os
import random
from sklearn.metrics import f1_score

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

CFG = {
    'NUM_WORKERS':0,
    'ANTIGEN_WINDOW':64,
    'ANTIGEN_MAX_LEN':64, # ANTIGEN_WINDOW와 ANTIGEN_MAX_LEN은 같아야합니다.
    'EPITOPE_MAX_LEN':25,
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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(CFG['SEED']) # Seed 고정

# Setup to initialize from the correctly named model files.
class mLSTMCell1900(tf.keras.Model):

    def __init__(self,
                 num_units,
                 model_path="./",
                 wn=True,
                 scope='mlstm',
                 var_device='cpu:0',
                 ):
        # Really not sure if I should reuse here
        super(mLSTMCell1900, self).__init__()
        self._num_units = num_units
        self._model_path = model_path
        self._wn = wn
        self._scope = scope
        self._var_device = var_device

        with tf.compat.v1.variable_scope(self._scope):
            self.wx_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wx.npy"))
            self.wh_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wh.npy"))
            self.wmx_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wmx.npy"))
            self.wmh_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wmh.npy"))
            self.b_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_b.npy"))
            self.gx_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gx.npy"))
            self.gh_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gh.npy"))
            self.gmx_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gmx.npy"))
            self.gmh_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gmh.npy"))
            self.wx = tf.compat.v1.get_variable(
                "wx", initializer=self.wx_init)
            self.wh = tf.compat.v1.get_variable(
                "wh", initializer=self.wh_init)
            self.wmx = tf.compat.v1.get_variable(
                "wmx", initializer=self.wmx_init)
            self.wmh = tf.compat.v1.get_variable(
                "wmh", initializer=self.wmh_init)
            self.b = tf.compat.v1.get_variable(
                "b", initializer=self.b_init)
            if self._wn:
                self.gx = tf.compat.v1.get_variable(
                    "gx", initializer=self.gx_init)
                self.gh = tf.compat.v1.get_variable(
                    "gh", initializer=self.gh_init)
                self.gmx = tf.compat.v1.get_variable(
                    "gmx", initializer=self.gmx_init)
                self.gmh = tf.compat.v1.get_variable(
                    "gmh", initializer=self.gmh_init)

    @property
    def state_size(self):
        # The state is a tuple of c and h
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        # The output is h
        return (self._num_units)

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units], dtype=dtype)
        h = tf.zeros([batch_size, self._num_units], dtype=dtype)
        return (c, h)

    def call(self, inputs, state):
        # Inputs will be a [batch_size, input_dim] tensor.
        # Eg, input_dim for a 10-D embedding is 10
        # nin = inputs.get_shape()[1].value

        # size will be (Batch_size, Seq_size, Input_size) (3-dim)
        # if not, (Seq_size, Input_size) (2-dim)

        # Unpack the state tuple
        if state is None:
            c_prev = tf.zeros((inputs.shape[0], 1900))
            h_prev = tf.zeros((inputs.shape[0], 1900))
        else:
            c_prev, h_prev = state

        if self._wn:
            wx = tf.nn.l2_normalize(self.wx, dim=0) * self.gx
            wh = tf.nn.l2_normalize(self.wh, dim=0) * self.gh
            wmx = tf.nn.l2_normalize(self.wmx, dim=0) * self.gmx
            wmh = tf.nn.l2_normalize(self.wmh, dim=0) * self.gmh
        else:
            wx = self.wx
            wh = self.wh
            wmx = self.wmx
            wmh = self.wmh

        m = tf.matmul(inputs, wmx) * tf.matmul(h_prev, wmh)
        z = tf.matmul(inputs, wx) + tf.matmul(m, wh) + self.b
        i, f, o, u = tf.split(z, 4, 1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c_prev + i * u
        h = o * tf.tanh(c)
        return h, (c, h)

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

all_df = pd.read_csv('./train.csv')
# Split Train : Validation = 0.8 : 0.2
# train_len = int(len(all_df)*0.8)
all_df = all_df[all_df['end_position'] - all_df['start_position'] + 1 <= 25]
train_positive_len = int(len(all_df[all_df['label']==1])*0.8)
train_negative_len = int(len(all_df[all_df['label']==0])*0.8)
train_positive = all_df[all_df['label']==1]
train_negative = all_df[all_df['label']==0]
train_df = train_positive.iloc[:train_positive_len]
val_df = train_positive.iloc[train_positive_len:]
train_df = train_df.append(train_negative[:train_negative_len], ignore_index=True)
val_df = val_df.append(train_negative[train_negative_len:], ignore_index=True)

train_epitope_list, train_left_antigen_list, train_right_antigen_list, train_total_antigen_list, train_label_list = get_custom_preprocessing('train', train_df)
val_epitope_list, val_left_antigen_list, val_right_antigen_list, val_total_antigen_list, val_label_list = get_custom_preprocessing('val', val_df)

class CustomDataset(Sequence):
    def __init__(self, epitope_list, left_antigen_list, right_antigen_list, total_antigen_list, label_list, batch_size=512):
        self.epitope_list = epitope_list
        self.left_antigen_list = left_antigen_list
        self.right_antigen_list = right_antigen_list
        self.total_antigen_list = total_antigen_list
        self.label_list = label_list
        self.batch_size = batch_size

    def __getitem__(self, index):
        self.epitope = self.epitope_list[index * self.batch_size : (index+1)*(self.batch_size)]
        self.left_antigen = self.left_antigen_list[index * self.batch_size : (index+1)*(self.batch_size)]
        self.right_antigen = self.right_antigen_list[index * self.batch_size : (index+1)*(self.batch_size)]
        self.total_antigen = self.total_antigen_list[index * self.batch_size : (index+1)*(self.batch_size)]

        if self.label_list is not None:
            self.label = self.label_list[index]
            return tf.convert_to_tensor(self.epitope), tf.convert_to_tensor(self.left_antigen), tf.convert_to_tensor(
                self.right_antigen), tf.convert_to_tensor(self.total_antigen), self.label
        else:
            return tf.Tensor(self.epitope), tf.Tensor(self.left_antigen), tf.Tensor(self.right_antigen), tf.Tensor(self.total_antigen)

    def __len__(self):
        return len(self.epitope_list)


class BaseModel(tf.keras.Model):
    def __init__(self,
                 epitope_length=CFG['EPITOPE_MAX_LEN'],
                 epitope_emb_node=10,
                 epitope_hidden_dim=1900,
                 total_hidden_dim=1900,
                 left_antigen_length=CFG['ANTIGEN_MAX_LEN'],
                 left_antigen_emb_node=10,
                 left_antigen_hidden_dim=64,
                 right_antigen_length=CFG['ANTIGEN_MAX_LEN'],
                 right_antigen_emb_node=10,
                 right_antigen_hidden_dim=64,
                 lstm_bidirect=False,
                 model_path = None
                 ):
        super(BaseModel, self).__init__()
        self.lstm_bidirect = lstm_bidirect
        self._model_path = model_path
        # Embedding Layer
        weights = np.load(os.path.join(self._model_path, "embed_matrix.npy"))
        self.total_embed = layers.Embedding(
            input_dim=27,
            output_dim=epitope_emb_node,
            mask_zero = True,
            weights = [np.vstack([weights, np.zeros_like(weights[0])])],
            trainable=False
        )

        self.rnn = mLSTMCell1900(2*CFG['ANTIGEN_MAX_LEN']+CFG['EPITOPE_MAX_LEN'],
                                model_path=self._model_path)

        # LSTM
        # self.forlstm = LSTMNet(model_path='1900_weights')
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
            # in_channels = epitope_hidden_dim + left_antigen_hidden_dim + right_antigen_hidden_dim
            in_channels = epitope_hidden_dim + total_hidden_dim

        self.classifier = tf.keras.Sequential([
            layers.LeakyReLU(True),
            layers.BatchNormalization(in_channels),
            layers.Dense(in_channels // 10),
            layers.LeakyReLU(True),
            layers.BatchNormalization(in_channels // 10),
            layers.Dense(1)
        ])

    def call(self, epitope_x, left_antigen_x, right_antigen_x, total_antigen_x):
        # BATCH_SIZE = epitope_x.size(0)
        # Get Lengths and Embedding Vector
        epitope_lengths = tf.reduce_sum(tf.cast(epitope_x != alpha_map['<PAD>'], tf.int32))
        # epitope is always >0
        left_lengths = tf.reduce_sum(tf.cast(left_antigen_x != alpha_map['<PAD>'], tf.int32))
        left_lengths += tf.where(left_lengths == 0, 1, 0)
        right_lengths = tf.reduce_sum(tf.cast(right_antigen_x != alpha_map['<PAD>'], tf.int32))
        right_lengths += tf.where(right_lengths == 0, 1, 0)
        total_lengths = left_lengths + epitope_lengths + right_lengths
        '''
        if self.lstm_bidirect:
            flipped_epitope_x = torch.stack(
                [torch.cat((torch.flip(epitope_x[t,:epitope_lengths[t]],dims=[0]), epitope_x[t, epitope_lengths[t]:])) for t in range(epitope_x.shape[0])]
            )
            flipped_epitope_x = self.total_embed(flipped_epitope_x)
        '''

        epitope_x = self.total_embed(epitope_x)
        total_antigen_x = self.total_embed(total_antigen_x)
        # left_antigen_x = self.total_embed(left_antigen_x)
        # right_antigen_x = self.total_embed(right_antigen_x)

        # LSTM
        # epitope_x = torch.nn.utils.rnn.pack_padded_sequence(epitope_x, epitope_lengths, batch_first=True,
        #                                                     enforce_sorted=False)
        state = None
        epitope_hidden_forward = tf.reshape(tf.constant([], dtype=tf.float32),(epitope_x.shape[0], 0, 1900))
        # epitope_hidden_forward = tf.experimental.numpy.empty(tf.zeros((epitope_x.shape[0], epitope_x.shape[1], 1900)))
        # epitope_hidden_forward = tf.Variable(tf.zeros((epitope_x.shape[0], epitope_x.shape[1], 1900)))
        for t in range(epitope_x.shape[0]):
            temp_hidden, state = self.rnn(epitope_x[:,t,:], state)
            # results = tf.stack([results, temp_hidden])
            # epitope_hidden_forward[:,t,:].assign(temp_hidden)
            epitope_hidden_forward = tf.concat([epitope_hidden_forward, tf.expand_dims(temp_hidden,1)], axis=1)

        epitope_hidden_forward = epitope_hidden_forward[tf.range(epitope_hidden_forward.shape[0]),
                                 (epitope_lengths - 1), :]

        state = None
        total_hidden_forward = tf.reshape(tf.constant([], dtype=tf.float32), (epitope_x.shape[0], 0, 1900))
        # total_hidden_forward = tf.Variable(tf.zeros((epitope_x.shape[0], epitope_x.shape[1], 1900)))
        for t in range(epitope_x.shape[0]):
            total_hidden, state = self.rnn(total_antigen_x[:,t,:], state)
            # total_hidden_forward[:,t,:].assign(total_hidden)
            total_hidden_forward = tf.concat([total_hidden_forward, tf.expand_dims(total_hidden,1)], axis=1)

        total_hidden_forward = total_hidden_forward[tf.range(total_hidden_forward.shape[0]),
                               (total_lengths - 1), :]

        epitope_hidden = epitope_hidden_forward
        epitope_hidden = tf.concat([total_hidden_forward, epitope_hidden], axis=-1)

        # Feature Concat -> Binary Classifier
        x = epitope_hidden
        # x = torch.cat([epitope_hidden, left_antigen_hidden, right_antigen_hidden], axis=-1)
        x = self.classifier(x).view(-1)
        return x

def train(model, optimizer, train_loader, val_loader):
    model.trainable = True
    counter = 0
    best_val_f1 = 0
    criterion = losses.BinaryCrossentropy()
    for epoch in range(1, CFG['EPOCHS'] + 1):
        train_loss = []
        for epitope_seq, left_antigen_seq, right_antigen_seq, total_antigen_seq, label in tqdm(iter(train_loader)):
            epitope_seq = epitope_seq
            left_antigen_seq = left_antigen_seq
            right_antigen_seq = right_antigen_seq
            total_antigen_seq = total_antigen_seq
            label = label

            with tf.GradientTape() as tape:
                output = model(epitope_seq, left_antigen_seq, right_antigen_seq, total_antigen_seq)
                loss = criterion(output, label)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            train_loss.append(loss)

        val_loss, val_f1 = validation(model, val_loader, criterion)
        print(
            f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val F1 : [{val_f1:.5f}]')

        if best_val_f1 < val_f1:
            best_val_f1 = val_f1
            tf.keras.models.save_model(model, './best_model_tf.pth')
            print('Model Saved.')
            counter=0
        else:
            counter +=1
            print('model not improved... counter ', counter)
            if counter >= CFG['PATIENCE']:
                print('early stopping')
                break
    return best_val_f1

def validation(model, val_loader, criterion):
    model.trainable = False
    pred_proba_label = []
    true_label = []
    val_loss = []
    for epitope_seq, left_antigen_seq, right_antigen_seq, total_antigen_seq, label in tqdm(iter(val_loader)):
        epitope_seq = epitope_seq
        left_antigen_seq = left_antigen_seq
        right_antigen_seq = right_antigen_seq
        total_antigen_seq = total_antigen_seq

        model_pred = model(epitope_seq, left_antigen_seq, right_antigen_seq, total_antigen_seq)
        loss = criterion(model_pred, label)
        model_pred = activations.sigmoid(model_pred)

        pred_proba_label += model_pred.numpy().tolist()
        true_label += label.numpy().tolist()

        val_loss.append(loss)

    pred_label = np.where(np.array(pred_proba_label) > CFG['THRESHOLD'], 1, 0)
    val_f1 = f1_score(true_label, pred_label, average='macro')
    return np.mean(val_loss), val_f1

train_loader = CustomDataset(train_epitope_list, train_left_antigen_list, train_right_antigen_list, train_total_antigen_list, train_label_list)
val_loader = CustomDataset(val_epitope_list, val_left_antigen_list, val_right_antigen_list, val_total_antigen_list, val_label_list)

model = BaseModel(model_path='1900_weights')
optimizer = optimizers.Adam()
best_score = train(model, optimizer, train_loader, val_loader)