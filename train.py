from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from trainer import train

from params import params
import os

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train(args, params)  ## Train.py file just calls train from learner.py to start training.


if __name__ == '__main__': ## As it is the file to run, we add this line and we ask the user for several arguments for the training
    parser = ArgumentParser(description='train (or resume training) a DiffWave model')
    parser.add_argument('--model-dir', default="./ckpt/",
        help='directory in which to store model checkpoints and training logs')
    parser.add_argument('--data-dir', default=None,
        help='directories from which to read .wav files for training')
    parser.add_argument('--max-steps', default=None, type= int,
        help='maximum number of training steps')
    parser.add_argument('--fp16',action='store_true', default=False,
        help='use 16-bit floating point operations for training')
    main(parser.parse_args())
