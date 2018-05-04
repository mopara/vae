import torch as T
from os.path import basename, join, splitext

def save(arr, file_name, dst_dir):
  T.save(T.tensor(arr), join(dst_dir, splitext(basename(file_name))[0]))
