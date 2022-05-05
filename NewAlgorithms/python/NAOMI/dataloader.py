
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

from helpers import draw_and_stats


def run_epoch(train, model, train_data, clip, optimizer=None, batch_size=25, teacher_forcing=True):

    losses = []

    i = 0
    next_i = batch_size
    while i < train_data.shape[0]:
        # print(f'{i=} {next_i=}')
        if next_i > train_data.shape[0]:
            next_i = train_data.shape[0]
        data = train_data[i:next_i].transpose(0, 1)
        # print(f'{data.shape=}')
        i = next_i
        next_i += batch_size

        has_value = Variable(~data.isnan())
        data = data.nan_to_num(nan=0.0)
        ground_truth = data.clone()
        data = torch.cat([has_value, data], 2)
        # print(f'{data.shape=}')
        seq_len = data.shape[0]

        if teacher_forcing:
            batch_loss = model(data, ground_truth)
        else:
            data_list = []
            for j in range(seq_len):
                data_list.append(data[j:j+1])
            samples = model.sample(data_list)
            batch_loss = torch.mean((ground_truth - samples).pow(2))
            batch_loss = Variable(batch_loss, requires_grad=True)

        if train:
            optimizer.zero_grad()
            total_loss = batch_loss
            # total_loss = Variable(batch_loss.data, requires_grad=True)
            total_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

        losses.append(batch_loss.data.cpu().numpy())
    return np.mean(losses)

def collect_samples_interpolate(policy_net, train_data, use_gpu, i_iter, size=25, name="sampling_inter", draw=False, stats=False, num_missing=None):
    total_samples = []
    total_states = []
    total_actions = []
    total_exp_states = []
    total_exp_actions = []
    total_res = []

    i = 0
    next_i = size
    while i < train_data.shape[0]:
        # print(f'{i=}')
        if next_i > train_data.shape[0]:
            next_i = train_data.shape[0]
        data = train_data[i:next_i].clone()
        # print(f'{data.shape=}')
        seq_len = data.shape[1]
        if use_gpu:
            data = data.cuda()
        data = Variable(data.transpose(0,1))
        _, missing_list, _ = torch.where(train_data.isnan())
        has_value = Variable(~data.isnan())
        data = data.nan_to_num(nan=0.0)
        ground_truth = data.clone()
        data = torch.cat([has_value, data], 2)
        data_list = []
        for j in range(seq_len):
            data_list.append(data[j:j+1])
        samples = policy_net.sample(data_list)
        states = samples[:-1, :, :]
        actions = samples[1:, :, :]
        exp_states = ground_truth[:-1, :, :]
        exp_actions = ground_truth[1:, :, :]
        res = samples.data.cpu()
        if i == 0:
            total_samples = samples
            total_states = states
            total_actions = actions
            total_exp_states = exp_states
            total_exp_actions = exp_actions
            total_res = res
        else:
            # print(f'{total_res.shape=}')
            total_samples = torch.cat([total_samples, samples], 1)
            total_states = torch.cat([total_states, states], 1)
            total_actions = torch.cat([total_actions, actions], 1)
            total_exp_states = torch.cat([total_exp_states, exp_states], 1)
            total_exp_actions = torch.cat([total_exp_actions, exp_actions], 1)
            total_res = torch.cat([total_res, res], 1)

        i = next_i
        next_i += size

        # print(f'{res.squeeze().shape=}')
        # np.savetxt("imgs/{}_{}".format(name, i_iter), res.squeeze())


        # mod_stats = draw_and_stats(samples.data, name + '_' + str(num_missing), i_iter, task, draw=draw,
        #                            compute_stats=stats, missing_list=missing_list)
        # exp_stats = draw_and_stats(ground_truth.data, name + '_expert' + '_' + str(num_missing), i_iter, task, draw=draw,
        #                            compute_stats=stats, missing_list=missing_list)

    return total_exp_states.data, total_exp_actions.data, train_data.data, total_states, total_actions, total_samples.data, 0, 0


def validation(policy_net, validation_data, use_gpu, size=25):
    total_res = []
    i = 0
    next_i = size
    while i < validation_data.shape[0]:
        # print(f'{i=}')
        if next_i > validation_data.shape[0]:
            next_i = validation_data.shape[0]
        data = validation_data[i:next_i].clone()
        # print(f'{data.shape=}')
        seq_len = data.shape[1]
        data = Variable(data.transpose(0, 1))
        if use_gpu:
            data = data.cuda()
        has_value = Variable(~data.isnan())
        data = data.nan_to_num(nan=0.0)
        ground_truth = data.clone()
        data = torch.cat([has_value, data], 2)
        data_list = []
        for j in range(seq_len):
            data_list.append(data[j:j + 1])
        samples = policy_net.sample(data_list)
        res = samples.data.cpu()
        if i == 0:
            total_res = res
        else:
            total_res = torch.cat([total_res, res], 1)

        i = next_i
        next_i += size
    return total_res.squeeze()