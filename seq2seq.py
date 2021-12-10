import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class Encoder(nn.Module):
  def __init__(self, input_feature_size, hidden_size: int):
    super(Encoder, self).__init__()
    self.lstm = nn.LSTM(input_feature_size, hidden_size, batch_first=True)
  
  def forward(self, x: torch.Tensor):
    # x: (BATCH, ARRAY_LEN, input_feature)
    return self.lstm(x)


class Decoder(nn.Module):
    def __init__(self, input_feature_size, hidden_size, choice_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_feature_size, hidden_size, batch_first=True)
        self.dense_layer = nn.Linear(hidden_size, choice_size)

    def forward(self, x, hidden, cell):
        # x: (BATCH, 1, input_feature) -> decode one by one
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = F.relu(out)
        out = self.dense_layer(out)

        return out, hidden, cell


class Seq2Seq(nn.Module):
    DECODER_INPUT_FEATURE_SIZE = 1

    def __init__(self, enc_input_feature_size, hidden_size, choice_size):
        super(Seq2Seq, self).__init__()

        self._choice_size = choice_size

        self.encoder = Encoder(enc_input_feature_size, hidden_size)
        self.decoder = Decoder(Seq2Seq.DECODER_INPUT_FEATURE_SIZE, hidden_size, choice_size)

    def forward(self, x, y, teacher_force_ratio=0.0):
        batch_size = x.shape[0]

        _, (hidden, cell) = self.encoder.forward(x)

        loss = 0

        dec_in = torch.zeros((batch_size, 1, 1)).cuda()
        cell = torch.zeros(cell.shape).cuda()

        preds = []

        for i in range(self._choice_size):
            out, hidden, cell = self.decoder(dec_in, hidden, cell)
            prob = F.softmax(out, dim=2)
            loss += F.cross_entropy(torch.squeeze(prob, 1), y[:, i])
            dec_in = torch.unsqueeze(prob.argmax(2), -1).float()

            is_teacher = random.random() < teacher_force_ratio
            dec_in = y[:, i].unsqueeze(1).unsqueeze(1).float() if is_teacher else dec_in

            preds.append(dec_in)

        preds = torch.cat(preds, dim=1)
        return preds, loss / batch_size / self._choice_size
