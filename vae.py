import argparse
import numpy as np

import torch as T
import torch.utils.data as data

import models

# todo: make sure everything runs on gpu

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=1, type=int, metavar="S",
  help="random seed (default: 1)")
parser.add_argument("--no-cuda", action="store_true", default=False,
  help="enable CUDA training")
parser.add_argument("data", type=str, help="input data", metavar="X")
parser.add_argument("--batch-size", default=128, type=int,
  help="input batch size for training (default: 128)", metavar="N")
parser.add_argument("--shuffle", action="store_true", default=False,
  help="shuffle training data (default: false")
parser.add_argument("--num-epochs", default=10, type=int,
  help="number of epochs to train (default: 10)", metavar="E")

def train(model, num_epochs, batches):
  model.train()

  for epoch in xrange(num_epochs):
    epoch_loss = 0

    for (X, Y) in batches:
      batch_size = X.shape[0]

      Y_prd = model(X)

      batch_loss = model.loss(Y_prd, Y)
      # batch_loss is the average loss per observation in a batch
      # batch_loss * batch_size will be the total loss over the observations in the batch
      epoch_loss += batch_loss * batch_size
      # epoch_loss += batch_loss

      model.opt.zero_grad()
      batch_loss.backward()
      model.opt.step()

    fmt = "Epoch: %03d\tLoss: %g"
    args = (epoch, epoch_loss)

    print fmt % args

def get_batches(file_name, batch_size, shuffle):
  X_trn = T.load(file_name)
  shape = X_trn.shape
  X_trn = X_trn.reshape(shape[0], -1).float() / 255
  dataset = data.TensorDataset(X_trn, X_trn)
  batches = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

  return (X_trn, shape, batches)

if __name__ == "__main__":
  args = parser.parse_args()

  T.manual_seed(args.seed)

  device = T.device("cpu" if args.no_cuda else "cuda")

  X_trn, (_, height, width), batches = get_batches(args.data, args.batch_size,
    args.shuffle)

  ae = models.AE1(height*width, 32, height*width)

  train(ae, args.num_epochs, batches)

  print X_trn.shape
