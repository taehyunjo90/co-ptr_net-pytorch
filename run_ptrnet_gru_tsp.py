import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils


from model.pointer_network_gru import PtrNetGRU
from data.tsp_data import tsp_iterator


def get_distance(points, answer):
    answer = answer.reshape(-1)

    if len(set([a.item() for a in answer])) != 5:
        length = 10.0
    else:
        length = 0
        for i in range(len(answer) - 1):
            a_point = points[int(answer[i])]
            b_point = points[int(answer[i + 1])]
            l = sum((a_point - b_point) ** 2) ** (1/2)
            length += l

    return length


if __name__ == "__main__":
    input_feature_size = 2
    choice_size = 6
    attention_size = 64
    hidden_size = 512

    ptr_net = PtrNetGRU(input_feature_size, hidden_size, attention_size, choice_size).cuda()

    optimizer = optim.Adam(ptr_net.parameters())

    for p in ptr_net.parameters():
        nn.init.uniform_(p, -0.08, 0.08)
    
    losses = []

    tsp_train_iterator = tsp_iterator(128, is_train=True)
    tsp_test_iterator = tsp_iterator(1024, is_train=False)

    test_set_show_count = 5

    for i, (x_batch, y_batch) in enumerate(tsp_train_iterator):
        ptr_net.train()
        optimizer.zero_grad()

        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

        preds, loss = ptr_net.forward(x_batch, y_batch, 1.0)

        losses.append(loss)

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(sum(losses) / len(losses))

            x_batch, y_batch = next(tsp_test_iterator)

            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            with torch.no_grad():
                ptr_net.eval()
                preds, loss = ptr_net.forward(x_batch, y_batch, 0.0)

            for i in range(test_set_show_count):
                pred_tsp_len = get_distance(x_batch[i], preds[i])
                real_tsp_len = get_distance(x_batch[i], y_batch[i])
                print(f"prediction tsp length: {pred_tsp_len}, optimal tsp length: {real_tsp_len}, diff: {pred_tsp_len - real_tsp_len}")
            print("-----------------------------")

            losses = []



