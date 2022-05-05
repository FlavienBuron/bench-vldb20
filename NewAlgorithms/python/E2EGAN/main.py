from __future__ import print_function
import time

from torch import seed

import models.e2egan as E2EGAN
import tensorflow as tf
import argparse
import numpy as np
import random
from data_loader import DataLoader
import os

SEED = 1
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--gen-length', type=int, default=96)
    parser.add_argument('--impute-iter', type=int, default=400)
    parser.add_argument('--pretrain-epoch', type=int, default=5)
    parser.add_argument('--run-type', type=str, default='train')
    parser.add_argument('--data-path', type=str, default="../set-a/")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--g_loss_lambda',type=float,default=100)
    parser.add_argument('--beta1',type=float,default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--n-inputs', type=int, default=1)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--z-dim', type=int, default=64)
#     parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
#                         help='Directory name to save the checkpoints')
#     parser.add_argument('--result-dir', type=str, default='results',
#                         help='Directory name to save the generated images')
#     parser.add_argument('--log-dir', type=str, default='logs',
#                         help='Directory name to save training logs')
    parser.add_argument('--isNormal',type=int,default=1)
    #0 false 1 true
    parser.add_argument('--isBatch-normal',type=int,default=0)
    parser.add_argument('--isSlicing',type=int,default=1)
    parser.add_argument('--disc_iters',type=int,default=8)

    parser.add_argument('--input',type=str)
    parser.add_argument('--output',type=str)
    parser.add_argument('--runtime', type=int, default=0)
    args = parser.parse_args()
    
    if args.isBatch_normal==0:
            args.isBatch_normal=False
    if args.isBatch_normal==1:
            args.isBatch_normal=True
    if args.isNormal==0:
            args.isNormal=False
    if args.isNormal==1:
            args.isNormal=True
    if args.isSlicing==0:
            args.isSlicing=False
    if args.isSlicing==1:
            args.isSlicing=True
    
    input_matrix = np.loadtxt(args.input)
    row, col = input_matrix.shape
    args.shape = (row, col)

    epochs=[args.epoch]
    g_loss_lambdas=[args.g_loss_lambda]
    disc_iters = [7]
    for disc in disc_iters:
        for e in epochs:
            for g_l in g_loss_lambdas:
                args.epoch=e
                args.disc_iters = disc
                args.g_loss_lambda=g_l
                tf.reset_default_graph()
                tf.set_random_seed(SEED)
                dt_train=DataLoader(input_matrix)
                tf.reset_default_graph()
                config = tf.ConfigProto() 
                config.gpu_options.allow_growth = True
                start_time = time.time()
                with tf.Session(config=config) as sess:
                    gan = E2EGAN.E2EGAN(sess,
                                args=args,
                                datasets=dt_train,
                                )
            
                    # build graph
                    gan.build_model()
            
                    # show network architecture
                    #show_all_variables()
            
                    # launch the graph in a session
                    gan.train()
                    
                    print(" [*] Training finished!")

                    print(" [*] Train dataset Imputation begin!")
                    imputed_matrix = gan.imputation(dt_train, 1)
                    print(" [*] Train dataset Imputation finished!")

                
                end_time = time.time()
                exec_time = (end_time - start_time) * 1000 * 1000
                print("Time", "E2EGAN", ":", exec_time)

                if args.runtime > 0:
                    np.savetxt(args.output, np.array([exec_time]))
                else:
                    np.savetxt(args.output, imputed_matrix)


                tf.reset_default_graph()

if __name__ == '__main__':
    main()