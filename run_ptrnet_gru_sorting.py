import torch
import torch.optim as optim

from model.pointer_network_gru import PtrNetGRU
from data.sort_data import fixed_batch


if __name__ == "__main__":
    is_cuda = True

    input_feature_size = 1
    batch_size = 128
    attention_unit = 10
    hidden_size = 512
    seq_len = 10

    ptrnet_gru = PtrNetGRU(input_feature_size, hidden_size, attention_unit)
    if is_cuda:
        ptrnet_gru = ptrnet_gru.cuda()

    optimizer = optim.Adam(ptrnet_gru.parameters())
    losses = []

    for i in range(10000):
        ptrnet_gru.train()
        optimizer.zero_grad()

        x_batch, y_batch = fixed_batch(batch_size, seq_len)
        x_batch = torch.unsqueeze(x_batch, -1).float()
        if is_cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        preds, loss = ptrnet_gru.forward(x_batch, y_batch, 0.5)

        losses.append(loss)

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"trainig loss : {sum(losses) / len(losses)}")

            with torch.no_grad():
                ptrnet_gru.eval()
                x_batch, y_batch = fixed_batch(batch_size, seq_len)
                x_batch = torch.unsqueeze(x_batch, -1).float()
                if is_cuda:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()

                preds, loss = ptrnet_gru.forward(x_batch, y_batch, 0.0)

            print(sum(preds == y_batch) / len(y_batch))

            losses = []







