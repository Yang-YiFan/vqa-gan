from trainer import Trainer
import argparse
from PIL import Image
import os

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str, default='./datasets',
                    help='input directory for visual question answering.')

parser.add_argument('--log_dir', type=str, default='./logs',
                    help='directory for logs.')

parser.add_argument('--model_dir', type=str, default='./models',
                    help='directory for saved models.')

parser.add_argument('--max_qst_length', type=int, default=30,
                    help='maximum length of question. \
                          the length in the VQA dataset = 26.')

parser.add_argument('--max_num_ans', type=int, default=10,
                    help='maximum number of answers.')

parser.add_argument('--embed_size', type=int, default=1024,
                    help='embedding size of feature vector \
                          for both image and question.')

parser.add_argument('--word_embed_size', type=int, default=300,
                    help='embedding size of word \
                          used for the input in the LSTM.')

parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers of the RNN(LSTM).')

parser.add_argument('--hidden_size', type=int, default=512,
                    help='hidden_size in the LSTM.')

parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate for training.')

parser.add_argument('--step_size', type=int, default=10,
                    help='period of learning rate decay.')

parser.add_argument('--gamma', type=float, default=0.1,
                    help='multiplicative factor of learning rate decay.')

parser.add_argument('--num_epochs', type=int, default=30,
                    help='number of epochs.')

parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size.')

parser.add_argument('--num_workers', type=int, default=0,
                    help='number of processes working on cpu.')

parser.add_argument('--save_step', type=int, default=1,
                    help='save step of model.')

parser.add_argument("--inference", default=False, action='store_true')

parser.add_argument("--l1_coef", default=50, type=float)

parser.add_argument("--l2_coef", default=100, type=float)

parser.add_argument("--save_path", default='')

args = parser.parse_args()

trainer = Trainer(args)

if not args.inference:
    trainer.train()
else:
    trainer.predict()

