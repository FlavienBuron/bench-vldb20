pip2 install numpy scipy pandas sklearn
pip2 install tensorflow==1.14.0
pip2 install torch torchvision
pip2 install pandas
pip2 install ujson
pip2 install ipdb
pip2 install scikit-learn
pip2 install future

# python3 dependencies. 2 different versions of python 3 are used to fight compatibilities issues between the different
# Tensorflow/Keras version needed by the different models
sudo apt-get update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get install python3.6
sudo apt install python3.8
sudo apt install python3-venv python3-pip
# python 3.6 packages
python3.6 -m pip install torch
python3.6 -m pip install numpy
python3.6 -m pip install tensorflow==1.7.0
# python3.8 packages
python3.8 -m pip install tensorflow
python3.8 -m pip install numpy