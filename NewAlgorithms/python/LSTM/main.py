import time
import argparse

import torch
import torch.optim as optim
import numpy as np

from data_prep import data_prep
import data_loader
import utils
import lstm

BATCH_SIZE = 8
EPOCHS = 100

parser = argparse.ArgumentParser()
parser.add_argument('--input', type = str)
parser.add_argument('--output', type = str)
parser.add_argument('--runtime', type = int, default = 0)
args = parser.parse_args()


def train(model, input):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    data_iter = data_loader.get_loader(input, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            # print '\r {}'.format(idx)
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer)

            run_loss += ret['loss'].data

            print '\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx+1) * 100.0 / len(data_iter), run_loss / (idx+1.0)),
        #end for
    #end for

    return (model, data_iter)
#end func

def evaluate(model, val_iter):
    model.eval()

    imputations = []
    imputations2 = []
    evals = []

    for _, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        # eval_masks = ret['eval_masks'].data.cpu().numpy()
        # eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()
        imputations += imputation.tolist()
        # imputations2 += imputation[np.where(eval_masks == 1)].tolist()
        # evals += eval_[np.where(eval_masks == 1)].tolist()
    #end for
    # evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    # mae = np.abs(evals - imputations2).mean()
    # mre = np.abs(evals - imputations2).sum() / np.abs(evals).sum()

    # print 'MAE', mae
    # print 'MRE', mre

    return imputations
#end func

def run(input, output, rt = 0):
    matrix = np.loadtxt(input)
    n = len(matrix)
    # n = 1000
    data_prep(input, input + ".tmp")

    start = time.time()
    model = lstm.LSTM(n)

    if torch.cuda.is_available():
        model = model.cuda()

    (model, data_iter) = train(model, input + ".tmp")
    res = evaluate(model, data_iter)
    end = time.time()

    if rt > 0:
        np.savetxt(output, np.array([(end - start) * 1000 * 1000]))
    else:
        res = res[0,:n]
        res = res.reshape(n)
        matrix[:, 0] = res
        np.savetxt(output, matrix)

    print ''
    print 'Time (LSTM):', ((end - start) * 1000 * 1000)

if __name__ == '__main__':
    input = args.input
    output = args.output
    run(input, output, args.runtime)

