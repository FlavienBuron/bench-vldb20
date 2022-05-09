# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

from data_loader import data_loader
from gain import gain
from utils import rmse_loss


def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  input = args.input
  output = args.output
  rt = args.runtime
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness
  miss_data_x, data_m = data_loader(input)
  batch_size = miss_data_x.shape[1]//10
  gain_parameters['batch_size'] = batch_size
  print('Batch size: ', gain_parameters["batch_size"])

  # beginning of imputation process - start time measurement
  start_time = time.time()
  
  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters)

  # imputation is complete - stop time measurement
  end_time = time.time()

  # calculate the time elapsed in [!] microseconds
  exec_time = (end_time - start_time) * 1000 * 1000

  if rt > 0:
    np.savetxt(output, np.array([exec_time]))
  else:
    np.savetxt(output, imputed_data_x)

  # Report the RMSE performance
  # rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
  
  # print()
  # print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  # return imputed_data_x, rmse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      type=str)
  parser.add_argument(
      '--output',
      type=str)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  parser.add_argument(
      '--runtime',
      default=0,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  # imputed_data, rmse = main(args)
  main(args)
