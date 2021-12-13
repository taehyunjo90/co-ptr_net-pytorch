import torch


def sample(min_length=15, max_length=20):
    array_len = torch.randint(low=min_length, high=max_length + 1, size=(1,))
    x = torch.randint(high=50, size=(array_len,))
    y = x.argsort()
    return x, y


def batch(batch_size, min_len=15, max_len=20):
    array_len = torch.randint(low=min_len, high=max_len + 1, size=(1,))
    x = torch.randint(high=50, size=(batch_size, array_len))
    y = x.argsort(dim=1)
    return x, y


def fixed_sample(fixed_size):
    return sample(fixed_size, fixed_size)


def fixed_batch(batch_size, fixed_size):
    return batch(batch_size, fixed_size, fixed_size)


if __name__ == '__main__':
    for _ in range(10):
        x, arg = fixed_sample(5)
        print(x, arg, x[arg])


