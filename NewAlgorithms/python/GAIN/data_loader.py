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

'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np


def data_loader (data_name):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  miss_data_x = np.loadtxt(data_name, delimiter=" ")

  data_m = ~np.isnan(miss_data_x)
      
  return miss_data_x, data_m
