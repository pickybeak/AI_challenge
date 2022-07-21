import torch
import numpy as np
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class mLSTMCell(torch.nn.Module):
    def __init__(self, model_path=None, input_size=10, hidden_size=1900):
        super(mLSTMCell, self).__init__()
        self._model_path = model_path
        self.hidden_size = hidden_size
        self.input_size = input_size
        if self._model_path is not None:
            self.wx_init = torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wx.npy"))).to(device)
            self.wh_init = torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wh.npy"))).to(device)
            self.wmx_init = torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wmx.npy"))).to(device)
            self.wmh_init = torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wmh.npy"))).to(device)
            self.b_init = torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_b.npy"))).to(device)
            self.gx_init = torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gx.npy"))).to(device)
            self.gh_init = torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gh.npy"))).to(device)
            self.gmx_init = torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gmx.npy"))).to(device)
            self.gmh_init = torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gmh.npy"))).to(device)
            '''
            self.register_buffer('wx_init',
                                 torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wx.npy"))))
            self.register_buffer('wh_init',
                                 torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wh.npy"))))
            self.register_buffer('wmx_init',
                                 torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wmx.npy"))))
            self.register_buffer('wmh_init',
                                 torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wmh.npy"))))
            self.register_buffer('b_init',
                                 torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_b.npy"))))
            self.register_buffer('gx_init',
                                 torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gx.npy"))))
            self.register_buffer('gh_init',
                                 torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gh.npy"))))
            self.register_buffer('gh_init',
                                 torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gh.npy"))))
            self.register_buffer('gmx_init',
                                 torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gmx.npy"))))
            self.register_buffer('gmh_init',
                                 torch.Tensor(np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gmh.npy"))))
                                 
            '''
            self.wx = torch.nn.functional.normalize(self.wx_init, p=2, dim=0) * self.gx_init
            self.wh = torch.nn.functional.normalize(self.wh_init, p=2, dim=0) * self.gh_init
            self.wmx = torch.nn.functional.normalize(self.wmx_init, p=2, dim=0) * self.gmx_init
            self.wmh = torch.nn.functional.normalize(self.wmh_init, p=2, dim=0) * self.gmh_init
        else:
            ValueError()
        # load only

    def forward(self, x, state=None):
        bs = x.shape[0]
        if state is None:
            c_prev, h_prev = (
                torch.zeros(bs, self.hidden_size, device=device),
                torch.zeros(bs, self.hidden_size, device=device)
            )
        else:
            c_prev, h_prev = state
        m  = torch.matmul(x, self.wmx) * torch.matmul(h_prev, self.wmh)
        z = torch.matmul(x, self.wx) + torch.matmul(m, self.wh) + self.b_init
        i, f, o, u = torch.split(z, z.shape[1]//4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c = f * c_prev + i * u
        h = o * torch.tanh(c)
        return h, (c, h)

class LSTMNet(torch.nn.Module):
    def __init__(self, model_path=None, lstm_cell=None, input_size=10, hidden_size=1900):
        super(LSTMNet, self).__init__()
        self.model_path = model_path
        self.hidden_size = hidden_size
        self.input_size = input_size
        if lstm_cell is None:
            self.lstm_cell = mLSTMCell(model_path=model_path, input_size=input_size, hidden_size=hidden_size)
        else:
            self.lstm_cell = lstm_cell

    def forward(self, x):
        seq_sz = x.shape[1]
        state = None
        output = torch.zeros(x.shape[0], seq_sz, self.hidden_size, device=device)
        for t in range(seq_sz):
            if torch.sum(x[:, t, :]) == 0:
                break
            hidden, state = self.lstm_cell(x[:, t, :], state)
            output[:,t,:] = hidden

        return output, state