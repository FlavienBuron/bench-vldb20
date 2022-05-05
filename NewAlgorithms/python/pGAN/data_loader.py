
import numpy as np

import tools

def data_loader(filename):
    """
    :param filename: file to impute
    :return: dataX (file with missing values as 0.0), dataM (mask with 0 if missing else 1)
    """

    data = np.loadtxt(filename, delimiter=" ", skiprows=0)
    # transform reshape data into (nb_col, ts_len, 1)
    data = np.expand_dims(data.T, 2)
    dataX = np.nan_to_num(data, nan=0.0)
    dataM = (~np.isnan(data)).astype(np.float32)

    return dataX, dataM


# if __name__ == "__main__":
#     data_loader(7, 7)