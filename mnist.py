# http://yann.lecun.com/exdb/mnist/
import argparse
import numpy as np
import struct
import sys
import util

parser = argparse.ArgumentParser()

parser.add_argument("-o", default="./", type=str, help="destination directory",
  metavar="dst_dir")
parser.add_argument("file_names", nargs="+", type=str,
  help="input files")

# https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
def save_imgs(file_names, dst_dir):
  for file_name in file_names:
    with open(file_name, "rb") as file:
      # >: big-endian
      # H: ushort (2 bytes)
      # B: uchar (1 byte)
      # I: uint (4 bytes)
      _zeros, _dtype, dims = struct.unpack(">HBB", file.read(4))
      shape = [struct.unpack(">I", file.read(4))[0] for dim in range(dims)]
      arr = np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)

      util.save(arr, file_name, dst_dir)

if __name__ == "__main__":
  args = parser.parse_args()

  save_imgs(args.file_names, args.o)
