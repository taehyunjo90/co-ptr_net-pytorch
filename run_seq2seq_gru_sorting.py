import torch
import torch.optim as optim

from model.seq2seq_gru import Seq2SeqGRU
from data.sort_data import fixed_batch


if __name__ == "__main__":
    input_feature_size = 1
    batch_size = 128
    seq_len = 8
    hidden_size = 512

    cuda = torch.device('cuda')
    seq2seq = Seq2SeqGRU(1, hidden_size, seq_len).cuda()

    optimizer = optim.Adam(seq2seq.parameters())

    seq2seq.train()
    
    losses = []

    for i in range(10000):
        optimizer.zero_grad()

        x_batch, y_batch = fixed_batch(batch_size, seq_len)
        x_batch = torch.unsqueeze(x_batch, -1).float().cuda()
        y_batch = y_batch.cuda()

        preds, loss = seq2seq.forward(x_batch, y_batch, 0.5)

        losses.append(loss)

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"trainig loss : {sum(losses) / len(losses)}")

            x_batch, y_batch = fixed_batch(batch_size, seq_len)
            x_batch = torch.unsqueeze(x_batch, -1).float().cuda()
            y_batch = y_batch.cuda()
            preds, loss = seq2seq.forward(x_batch, y_batch, 0.0)

            print(sum(preds.squeeze(2) == y_batch) / len(y_batch))

            losses = []







