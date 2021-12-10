import torch
import random
torch.set_printoptions(precision=4)

def get_tsp_data(path):
    with open(path, 'r') as fp:
        ls = fp.readlines()

    total_x = []
    total_y = []

    for l in ls:
        x = []
        y = []
        l = l.split(" ")

        is_x = True
        for i, e in enumerate(l):
            if e == "output" or e == '\n':
                is_x = False
                continue

            if is_x:
                if i % 2 == 0:
                    x.append([float(l[i]), float(l[i + 1])])
            else:
                y.append(int(e) - 1) # idx from zero
        
        total_x.append(x)
        total_y.append(y)

    return torch.Tensor(total_x), torch.LongTensor(total_y)


def tsp_iterator(batch_size, is_train=True):
    if is_train:
        total_x, total_y = get_tsp_data(r"co_data/tsp5.txt")
    else:
        total_x, total_y = get_tsp_data(r"co_data/tsp5_test.txt")
    total_len = total_x.size(0)

    while True:
        i = random.randint(0, total_len - batch_size)
        yield total_x[i:i+batch_size], total_y[i:i+batch_size]


if __name__ == '__main__':
    for x_batch, y_batch in tsp_iterator(5, is_train=False):
        print(x_batch, y_batch)
        break
    