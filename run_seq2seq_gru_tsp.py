import torch
import torch.optim as optim
import torch.nn.utils


from model.seq2seq_gru import Seq2SeqGRU
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
    hidden_size = 512

    seq2seq = Seq2SeqGRU(input_feature_size, hidden_size, choice_size).cuda()

    optimizer = optim.Adam(seq2seq.parameters())
    
    losses = []

    tsp_train_iterator = tsp_iterator(128, is_train=True)
    tsp_test_iterator = tsp_iterator(128, is_train=False)

    test_set_show_count = 5

    for i, (x_batch, y_batch) in enumerate(tsp_train_iterator):
        seq2seq.train()
        optimizer.zero_grad()

        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

        preds, loss = seq2seq.forward(x_batch, y_batch, 0.9)

        losses.append(loss)

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(sum(losses) / len(losses))

            x_batch, y_batch = next(tsp_test_iterator)

            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            with torch.no_grad():
                seq2seq.eval()
                preds, loss = seq2seq.forward(x_batch, y_batch, 0.0)

            for i in range(test_set_show_count):
                pred_tsp_len = get_distance(x_batch[i], preds[i])
                real_tsp_len = get_distance(x_batch[i], y_batch[i])
                print(f"prediction tsp length: {pred_tsp_len}, optimal tsp length: {real_tsp_len}, diff: {pred_tsp_len - real_tsp_len}")
            print("-----------------------------")

            losses = []



