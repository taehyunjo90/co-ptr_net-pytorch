import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class EncoderGRU(nn.Module):
  def __init__(self, input_feature_size, hidden_size: int):
    super(EncoderGRU, self).__init__()
    self.gru = nn.GRU(input_feature_size, hidden_size, batch_first=True)
  
  def forward(self, x: torch.Tensor):
    # x: (BATCH, ARRAY_LEN, input_feature)
    return self.gru(x)


class DecoderGRU(nn.Module):
    def __init__(self, input_feature_size, hidden_size, choice_size):
        super(DecoderGRU, self).__init__()
        self.gru = nn.GRU(input_feature_size, hidden_size, batch_first=True)
        self.dense_layer = nn.Linear(hidden_size, choice_size)

    def forward(self, x, hidden):
        # x: (BATCH, 1, input_feature) -> decode one by one
        out, hidden = self.gru(x, hidden)
        out = torch.tanh(out)
        out = self.dense_layer(out)

        return out, hidden


class Seq2SeqGRU(nn.Module):
    DECODER_INPUT_FEATURE_SIZE = 1

    def __init__(self, enc_input_feature_size, hidden_size, choice_size):
        super(Seq2SeqGRU, self).__init__()

        self._choice_size = choice_size

        self.encoder = EncoderGRU(enc_input_feature_size, hidden_size)
        self.decoder = DecoderGRU(Seq2SeqGRU.DECODER_INPUT_FEATURE_SIZE, hidden_size, choice_size)

    def forward(self, x, y, teacher_force_ratio=0.0):
        batch_size = x.shape[0]

        _, hidden = self.encoder.forward(x)

        loss = 0

        dec_in = torch.zeros((batch_size, 1, 1)).cuda()

        preds = []

        for i in range(self._choice_size):
            out, hidden = self.decoder(dec_in, hidden)
            prob = F.softmax(out, dim=2)
            loss += F.cross_entropy(torch.squeeze(prob, 1), y[:, i])
            dec_in = torch.unsqueeze(prob.argmax(2), -1).float()

            is_teacher = random.random() < teacher_force_ratio
            dec_in = y[:, i].unsqueeze(1).unsqueeze(1).float() if is_teacher else dec_in

            preds.append(dec_in)

        preds = torch.cat(preds, dim=1)
        return preds, loss / batch_size / self._choice_size
