"""
Generate random data for pointer network
"""
import torch
from torch.utils.data import Dataset


def sample(min_length=15, max_length=20):
  """
  Generates a single example for a pointer network. The example consist in a tuple of two
  elements. First element is an unsorted array and the second element 
  is the result of applying argsort on the first element
  """
  array_len = torch.randint(low=min_length, 
                            high=max_length + 1,
                            size=(1,))
  x = torch.randint(high=50, size=(array_len,))
  return x, x.argsort()


def batch(batch_size, min_len=15, max_len=20):
  array_len = torch.randint(low=min_len, 
                            high=max_len + 1,
                            size=(1,))

  x = torch.randint(high=50, size=(batch_size, array_len))
  return x, x.argsort(dim=1)


def fixed_sample(fixed_size):
  return sample(fixed_size, fixed_size)

def fixed_batch(batch_size, fixed_size):
  return batch(batch_size, fixed_size, fixed_size)


if __name__ == '__main__':
  for _ in range(10):
    x, arg = fixed_sample(5)
    print(x, arg, x[arg])
