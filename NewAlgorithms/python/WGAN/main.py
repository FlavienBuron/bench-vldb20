from __future__ import print_function

import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import models.WGAN_GRUI as WGAN_GRUI
import tensorflow as tf
import argparse
import numpy as np
from data_loader import DataLoader
import os

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--gen-length', type=int, default=96)
    parser.add_argument('--impute-iter', type=int, default=400)
    parser.add_argument('--pretrain-epoch', type=int, default=5)
    parser.add_argument('--g_loss_lambda',type=float,default=0.1)
    parser.add_argument('--beta1',type=float,default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--n_inputs', type=int, default=1)
    parser.add_argument('--n_hidden_units', type=int, default=64)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--isNormal',type=int,default=1)
    parser.add_argument('--isBatch_normal',type=int,default=1)
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
    batch_size = col // 10
    args.batch_size = batch_size
    args.shape = (row, col)
    print(f'Batch size: {batch_size}')

    epochs=[args.epoch]
    g_loss_lambdas=[args.g_loss_lambda]
    beta1s = [0.5]

    for beta1 in beta1s:
        for e in epochs:
            for g_l in g_loss_lambdas:
                args.epoch=e
                args.beta1 = beta1
                args.g_loss_lambda=g_l
                tf.reset_default_graph()
                dt_train=DataLoader(input_matrix)# dt_test=readTestData.ReadPhysionetData(os.path.join(args.data_path,"test"), os.path.join(args.data_path,"test","list.txt"),dt_train.maxLength,isNormal=args.isNormal,isSlicing=args.isSlicing)
                tf.reset_default_graph()
                config = tf.ConfigProto() 
                config.gpu_options.allow_growth = True
                start_time = time.time()
                with tf.Session(config=config) as sess:
                    gan = WGAN_GRUI.WGAN(sess,
                                args=args,
                                datasets=dt_train,
                                )

                    # build graph
                    print("Building model")
                    gan.build_model()
                    # launch the graph in a session
                    print("Starting training")
                    gan.train()
                    print(" [*] Training finished!")

                    print(" [*] Train dataset Imputation begin!")
                    imputed_matrix = gan.imputation(dt_train, True)
                    print(" [*] Train dataset Imputation finished!")

                    end_time = time.time()
                    exec_time = (end_time - start_time) * 1000 * 1000
                    print("Time", "WGAN", ":", exec_time)

                    if args.runtime > 0:
                        np.savetxt(args.output, np.array([exec_time]))
                    else:
                        np.savetxt(args.output, imputed_matrix)

                tf.reset_default_graph()
if __name__ == '__main__':
    main()