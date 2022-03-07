import torch
import torch.nn as nn

from torch.autograd import Variable

LSTM_HID_SIZE = 16
NUM_OF_FEATURES = 1

class LSTM(nn.Module):
    def __init__(self, seq_len):
        super(LSTM, self).__init__()
        self.seq_len = seq_len
        self.build()

    def build(self):
        self.lstm_cell = nn.LSTMCell(NUM_OF_FEATURES*2, LSTM_HID_SIZE)
        self.regression = nn.Linear(LSTM_HID_SIZE, NUM_OF_FEATURES)

    def forward(self, data):
        values = data["forward"]['values']
        masks = data["forward"]['masks']
        evals = data["forward"]['evals']
        eval_masks = data["forward"]['eval_masks']

        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], LSTM_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], LSTM_HID_SIZE)))
        x_h = self.regression(h)
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0

        imputations = []

        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]

            # estimate x_h(t) based on previous output h(t-1)
            x_h = self.regression(h)
            # establish complement depending if value is missing, then use x_h(t)
            x_c =  m * x +  (1 - m) * x_h

            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            inputs = torch.cat([x_c, m], dim = 1)

            h, c = self.lstm_cell(inputs, (h, c))

            imputations.append(x_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        return {'loss': x_loss / self.seq_len,'imputations': imputations,\
                'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret