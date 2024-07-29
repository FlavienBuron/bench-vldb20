pip2 install numpy scipy pandas
pip2 install scikit-learn
pip2 install tensorflow==1.14.0
pip2 install torch torchvision
pip2 install pandas
pip2 install ujson
pip2 install ipdb
pip2 install future

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.8 -y
sudo apt install python3.8-venv python3-venv python3-pip -y
sudo apt install python3.8-distutils -y
sudo apt-get install python3.8-dev -y
# python3.8 packages
# creates and enter the benchmark virtual environment 'bench-env'
python3.8 -m venv bench-env
. bench-env/bin/activate
# ~~~~~~~~~~~~~~~START bench-env~~~~~~~~~~~~~~~~~~~~~~~~
python3.8 -m pip install einops
python3.8 -m pip install numpy~=1.19
python3.8 -m pip install pandas
python3.8 -m pip install pytorch-lightning~=1.4
python3.8 -m pip install scikit-learn
python3.8 -m pip install scipy
python3.8 -m pip install tensorflow~=2.5
python3.8 -m pip install tensorboard
python3.8 -m pip install pypots
python3.8 -m pip install torch~=1.8
python3.8 -m pip install torch_geometric
python3.8 -m pip install torch_scatter
python3.8 -m pip install torch-sparse
python3.8 -m pip install torch_cluster
python3.8 -m pip install pyaml
python3.8 -m pip install torchcde~=0.2.5
python3.8 -m pip install torchmetrics~=0.5
deactivate

# ~~~~~~~~~~~~~~~END bench-env~~~~~~~~~~~~~~~~~~~~~~~~
