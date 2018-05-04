import argparse
import cv2
import numpy as np
import sys
import util

parser = argparse.ArgumentParser()

parser.add_argument("-o", default="./", type=str, help="destination directory",
  metavar="dst_dir")
parser.add_argument("file_names", nargs="+", type=str,
  help="input files")

# https://stackoverflow.com/a/42166299
def save_vids(file_names, dst_dir):
  for file_name in file_names:
    vid = cv2.VideoCapture(file_name)

    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid_arr = np.empty((num_frames, height, width, 3), dtype=np.uint8)

    for frame_num in xrange(num_frames):
      ret, frame = vid.read()

      if ret:
        vid_arr[frame_num] = frame
      else:
        print "error in %s: frame [%d/%d]" % (file_name, frame_num, num_frames)
        break

    vid.release()

    util.save(vid_arr, file_name, dst_dir)

    gb = float(vid_arr.nbytes)/(2**30)

    print "%s saved. shape: %s. %0.4ggb" % (file_name, vid_arr.shape, gb)

if __name__ == "__main__":
  args = parser.parse_args()

  save_vids(args.file_names, args.o)
