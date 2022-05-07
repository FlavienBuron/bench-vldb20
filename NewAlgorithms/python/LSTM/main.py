import time
import argparse

import torch
import torch.optim as optim
import numpy as np

from data_prep import data_prep
import data_loader
import utils
import lstm

# BATCH_SIZE = 8
EPOCHS = 100

parser = argparse.ArgumentParser()
parser.add_argument('--input', type = str)
parser.add_argument('--output', type = str)
parser.add_argument('--runtime', type = int, default = 0)
args = parser.parse_args()


def train(model, input, batch_size=8):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    data_iter = data_loader.get_loader(input, batch_size=batch_size)

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

    for _, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        imputation = ret['imputations'].data.cpu().numpy()
        imputations += imputation.tolist()
    #end for

    imputations = np.asarray(imputations)

    return imputations
#end func

def run(input, output, rt = 0):
    matrix = np.loadtxt(input)
    n, m = matrix.shape
    batch_size = m//10
    print 'Batch size: {}'.format(batch_size)
    data_prep(input, input + ".tmp")

    start = time.time()
    model = lstm.LSTM(n)

    if torch.cuda.is_available():
        model = model.cuda()

    (model, data_iter) = train(model, input + ".tmp", batch_size)
    res = evaluate(model, data_iter)
    end = time.time()

    if rt > 0:
        np.savetxt(output, np.array([(end - start) * 1000 * 1000]))
    else:
        for i in range(0, len(res)):
            res_l = res[i, :n];
            matrix[:, i] = res_l.reshape(n);
        np.savetxt(output, matrix)

    print ''
    print 'Time (LSTM):', ((end - start) * 1000 * 1000)

if __name__ == '__main__':
    input = args.input
    output = args.output
    run(input, output, args.runtime)

