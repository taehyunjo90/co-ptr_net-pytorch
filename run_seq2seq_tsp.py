import torch
import torch.optim as optim
import torch.nn.utils

from seq2seq import Seq2Seq
from tsp_data import tsp_iterator

if __name__ == "__main__":
    input_feature_size = 2
    choice_size = 6
    hidden_size = 512

    seq2seq = Seq2Seq(input_feature_size, hidden_size, choice_size).cuda()

    optimizer = optim.Adam(seq2seq.parameters())
    
    losses = []

    seq2seq.train()

    tsp_train_iterator = tsp_iterator(8, is_train=True)
    tsp_test_iterator = tsp_iterator(512, is_train=False)

    for i, (x_batch, y_batch) in enumerate(tsp_train_iterator):
        optimizer.zero_grad()

        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

        preds, loss = seq2seq.forward(x_batch, y_batch, 0.0)

        losses.append(loss)

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(sum(losses) / len(losses))

            x_batch, y_batch = next(tsp_test_iterator)

            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            with torch.no_grad():
                preds, loss = seq2seq.forward(x_batch, y_batch, 0.0)
            print(sum(preds.squeeze(2) == y_batch) / len(y_batch))

            losses = []



