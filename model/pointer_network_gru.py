import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from data.sort_data import fixed_batch


class EncoderPTRGRU(nn.Module):
  def __init__(self, input_feature_size, hidden_size: int):
    super(EncoderPTRGRU, self).__init__()
    self.gru = nn.GRU(input_feature_size, hidden_size, batch_first=True)
  
  def forward(self, x: torch.Tensor):
    # x: (BATCH, ARRAY_LEN, input_feature)
    out, hidden = self.gru(x)
    return out, hidden


class DecoderPTRGRU(nn.Module):
    def __init__(self, input_feature_size, hidden_size):
        super(DecoderPTRGRU, self).__init__()
        self.gru = nn.GRU(input_feature_size, hidden_size, batch_first=True)

    def forward(self, x, hidden):
        assert x.size(1) == 1
        # x: (BATCH, 1, input_feature) -> decode one by one
        out, hidden = self.gru(x, hidden)
        return out, hidden


class PtrNetGRU(nn.Module):
    def __init__(self, input_feature_size, hidden_size, attention_unit_size, choice_size):
        super(PtrNetGRU, self).__init__()

        self._choice_size = choice_size

        self.encoder = EncoderPTRGRU(input_feature_size, hidden_size)
        self.decoder = DecoderPTRGRU(input_feature_size, hidden_size)

        self.w1 = nn.Linear(hidden_size, attention_unit_size, bias=False)
        self.w2 = nn.Linear(hidden_size, attention_unit_size, bias=False)
        self.v = nn.Linear(attention_unit_size, 1, bias=False)

    def forward(self, x, y, teacher_force_ratio=0.0):
        batch_size = x.size(0)
        enc_out, hidden = self.encoder.forward(x)

        loss = 0

        dec_in = torch.full((batch_size, 1, x.size(2)), 0.0).cuda() # start_token

        preds = []

        for i in range(self._choice_size):
            dec_out, hidden = self.decoder(dec_in, hidden)

            prob = (self.w1(enc_out) + self.w2(dec_out))
            prob = self.v(torch.relu(prob))
            prob = F.softmax(prob.squeeze(-1), dim=1)
            
            loss += F.cross_entropy(prob, y[:, i])

            predictions = prob.argmax(1)
            is_teacher = random.random() < teacher_force_ratio
            idx = y[:, i] if is_teacher else predictions

            dec_in = torch.stack([x[b, idx[b]] for b in range(batch_size)])
            dec_in = dec_in.unsqueeze(1)

            preds.append(predictions.reshape(predictions.size(0), -1))

        preds = torch.cat(preds, dim=1)
        return preds, loss / batch_size / self._choice_size
