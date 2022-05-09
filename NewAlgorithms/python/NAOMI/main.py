import argparse
import time
import os

from model import Discriminator, num_trainable_params, NAOMI
from dataloader import run_epoch, collect_samples_interpolate, validation
from helpers import update_discrim, update_policy

import torch
from torch import nn
import numpy as np

Tensor = torch.DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--runtime', type=int, default=0)

parser.add_argument('-t', '--trial', type=int, default=1)
parser.add_argument('--model', type=str, default='NAOMI', help='NAOMI, SingleRes')
parser.add_argument('--y_dim', type=int, default=1)
parser.add_argument('--rnn_dim', type=int, default=300)
parser.add_argument('--dec1_dim', type=int, default=200)
parser.add_argument('--dec2_dim', type=int, default=200)
parser.add_argument('--dec4_dim', type=int, default=200)
parser.add_argument('--dec8_dim', type=int, default=200)
parser.add_argument('--dec16_dim', type=int, default=200)
parser.add_argument('--n_layers', type=int, required=False, default=2)
parser.add_argument('--seed', type=int, required=False, default=123)
parser.add_argument('--clip', type=int, required=False, help='gradient clipping', default=10)
parser.add_argument('--pre_start_lr', type=float, required=False, default=1e-3, help='pretrain starting learning rate')
parser.add_argument('--batch_size', type=int, required=False, default=25)
parser.add_argument('--save_every', type=int, required=False, default=50, help='periodically save model')
parser.add_argument('--pretrain', type=int, required=False, default=10,
                    help='num epochs to use supervised learning to pretrain')
parser.add_argument('--highest', type=int, required=False, default=8,
                    help='highest resolution in terms of step size in NAOMI')
parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')

parser.add_argument('--discrim_rnn_dim', type=int, required=False, default=128)
parser.add_argument('--discrim_layers', type=int, required=False, default=1)
parser.add_argument('--policy_learning_rate', type=float, default=3e-6,
                    help='policy network learning rate for GAN training')
parser.add_argument('--discrim_learning_rate', type=float, default=1e-3,
                    help='discriminator learning rate for GAN training')
parser.add_argument('--max_iter_num', type=int, default=10,
                    help='maximal number of main iterations (default: 60000)')
parser.add_argument('--log_interval', type=int, default=1, help='interval between training status logs (default: 1)')
parser.add_argument('--draw_interval', type=int, default=200,
                    help='interval between drawing and more detailed information (default: 50)')
parser.add_argument('--pretrain_disc_iter', type=int, default=10,
                    help="pretrain discriminator iteration (default: 2000)")
parser.add_argument('--save_model_interval', type=int, default=50, help="interval between saving model (default: 50)")

args = parser.parse_args()
if not torch.cuda.is_available():
    args.cuda = False


def adversarial_training(policy_net, discrim_net, train_data, pretrain_disc_iter, max_iter_num, use_gpu=False):

    optimizer_policy = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy_net.parameters()),
        lr=args.policy_learning_rate)
    optimizer_discrim = torch.optim.Adam(
        discrim_net.parameters(), lr=args.discrim_learning_rate)
    discrim_criterion = nn.BCELoss()
    if use_gpu:
        discrim_criterion = discrim_criterion.cuda()

    exp_p = []
    mod_p = []

    for i in range(pretrain_disc_iter):
        exp_states, exp_actions, exp_seq, model_states_var, model_actions_var, model_seq, mod_stats, exp_stats = \
            collect_samples_interpolate(policy_net, train_data, use_gpu, i, size=args.batch_size, name="pretraining", draw=False,
                                        stats=False)
        model_states = model_states_var.data
        model_actions = model_actions_var.data
        pre_mod_p, pre_exp_p = update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states,
                                              exp_actions, model_states, model_actions, i, dis_times=3.0,
                                              use_gpu=use_gpu, train=True)
        print(f'\r{i}, exp: {pre_exp_p}, mod: {pre_mod_p}')

        if pre_mod_p < 0.3:
            break

    for i_iter in range(max_iter_num):
        ts0 = time.time()
        exp_states, exp_actions, exp_seq, model_states_var, model_actions_var, model_seq, mod_stats, exp_stats = \
            collect_samples_interpolate(policy_net, train_data, use_gpu, i_iter, size=args.batch_size, draw=False, stats=False)
        model_states = model_states_var.data
        model_actions = model_actions_var.data

        ts1 = time.time()
        t0 = time.time()

        # update discriminator
        mod_p_epoch, exp_p_epoch = update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states,
                                                  exp_actions, model_states, model_actions, i_iter, dis_times=3.0,
                                                  use_gpu=use_gpu, train=True)
        exp_p.append(exp_p_epoch)
        mod_p.append(mod_p_epoch)

        # update policy network
        if i_iter > 3 and mod_p[-1] < 0.8:
            update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, model_states_var,
                          model_actions_var, i_iter, use_gpu)
        t1 = time.time()


def pretrain(policy_net, train_data, pretrain_epochs, lr, teacher_forcing=True):
    best_test_loss = 0

    for e in range(pretrain_epochs):
        epoch = e + 1
        # print("Epoch: {}".format(epoch))
        # draw and stats
        _, _, _, _, _, _, mod_stats, exp_stats = \
            collect_samples_interpolate(policy_net, train_data, False, e, size=args.batch_size, name='pretrain_inter', draw=True,
                                        stats=True)

        if epoch == pretrain_epochs // 2:
            lr = lr / 10

        # train
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, policy_net.parameters()),
            lr=lr)

        start_time = time.time()

        train_loss = run_epoch(True, policy_net, train_data, args.clip, optimizer, batch_size=args.batch_size,
                               teacher_forcing=teacher_forcing)

        print(f'\rPretrain Epoch: {epoch}, Train Loss: {train_loss:.8f}')


def run():
    train_data = torch.Tensor(np.loadtxt(args.input).T).unsqueeze(2)
    args.batch_size = train_data.shape[0] // 10

    params = {
        'batch': args.batch_size,
        'y_dim': args.y_dim,
        'rnn_dim': args.rnn_dim,
        'dec1_dim': args.dec1_dim,
        'dec2_dim': args.dec2_dim,
        'dec4_dim': args.dec4_dim,
        'dec8_dim': args.dec8_dim,
        'dec16_dim': args.dec16_dim,
        'n_layers': args.n_layers,
        'discrim_rnn_dim': args.discrim_rnn_dim,
        'discrim_num_layers': args.discrim_layers,
        'cuda': args.cuda,
        'highest': args.highest,
    }

    use_gpu = args.cuda

    # hyperparameters
    pretrain_epochs = args.pretrain
    clip = args.clip
    start_lr = args.pre_start_lr
    batch_size = args.batch_size
    save_every = args.save_every

    # manual seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        if use_gpu:
            torch.cuda.manual_seed_all(args.seed)

    # build model
    policy_net = eval(args.model)(params)
    discrim_net = Discriminator(params).double()
    if torch.cuda.is_available:
        if args.cuda:
            policy_net, discrim_net = policy_net.cuda(), discrim_net.cuda()
    params['total_params'] = num_trainable_params(policy_net)

    start_time = time.time()

    # Data

    pretrain(policy_net, train_data, pretrain_epochs, start_lr, True)
    adversarial_training(policy_net, discrim_net, train_data, args.pretrain_disc_iter, args.max_iter_num, False)
    out_matrix = validation(policy_net, train_data, use_gpu, size=args.batch_size)

    end_time = time.time()
    exec_time = (end_time - start_time) * 1000 * 1000

    print("Time Naomi: ", exec_time)

    if args.runtime > 0:
        np.savetxt(args.output, np.array([exec_time]))
    else:
        np.savetxt(args.output, out_matrix)


if __name__ == '__main__':
    run()
