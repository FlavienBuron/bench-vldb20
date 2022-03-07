import ujson as json
import numpy as np
import pandas as pd

def parse_delta(masks):
    deltas = []

    for t in range(len(masks)):
        if t == 0:
            deltas.append(np.ones(1))
        else:
            deltas.append(np.ones(1) + (1 - masks[t]) * deltas[-1])
    #end for
    return np.array(deltas)
#end func


def parse_rec(values, masks, evals, eval_masks):
    deltas = parse_delta(masks)
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values
    rec = {}

    rec["values"] = np.nan_to_num(values).tolist()
    rec["masks"] = masks.astype("int32").tolist()
    rec["evals"] = np.nan_to_num(evals).tolist()
    rec["eval_masks"] = eval_masks.astype("int32").tolist()
    rec["forwards"] = forwards.tolist()
    rec["deltas"] = deltas.tolist()

    return rec


def data_prep(input, output):

    file = open(output, 'w')
    src = pd.read_csv(input, header=None, delimiter=" ")

    for col in range(src.shape[1]):

        evals = []
        values = []

        for v in src[col].tolist():
            evals.append([float(v)])
    
        evals = np.array(evals)
        shp = evals.shape
        evals = evals.reshape(-1)
        values = evals.copy()
    
        masks = ~np.isnan(values)
        eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

        evals = evals.reshape(shp)
        values = values.reshape(shp)
        masks = masks.reshape(shp)
        eval_masks = eval_masks.reshape(shp)

        rec = {"label": 0}

        rec["forward"] = parse_rec(values, masks, evals, eval_masks)

        rec = json.dumps(rec)
        file.write(rec + '\n')

if __name__ == "__main__":
    out = open("./data/in/chlorine_normal.json", "w")
    data = pd.read_csv("./data/in/chlorine_normal.txt", header=None, delimiter=" ")
    for col in range(data.shape[1]):
        data_prep(data[col].tolist(), col, out)
    out.close()
