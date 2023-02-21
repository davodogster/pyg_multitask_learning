curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh

sha256sum Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh

cd ~/Documents/tree_segmentation/torch-points3d

## on R2D2

# source /home/scion.local/davidsos/anaconda3/bin/activate
conda activate points3d
# pip install torch # (1.9.0)
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-points3d
ssl._create_default_https_context = ssl._create_unverified_context



# python -m ipykernel install --user --name=TORCH3D
# Installed kernelspec TORCH3D in /home/scion.local/davidsos/.local/share/jupyter/kernels/torch3d
## WORKED!! followed this -https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084
## need conda forge not just lazy conda install.
# (points3d) davidsos@C26789:~/Documents/tree_segmentation/torch-points3d$ ipython kernel install --user --name=points4d
# Installed kernelspec points4d in /home/scion.local/davidsos/.local/share/jupyter/kernels/points4d

#################################################################
conda activate points3d

python -m unittest -v # 156 tests, 9 errors
# Ran 156 tests in 96.119s

# FAILED (errors=9) 

# Ran 166 tests in 25.577s FOR NIKOGAMULIN VERSION THOUGH!!

# FAILED (failures=4, errors=22) # worse than the 9 errors previously..

poetry run python train.py task=segmentation model_type=pointnet2 model_name=pointnet2_charlesssg dataset=shapenet-fixed
poetry run python train.py task=regression models=segmentation/rsconv model_name=pointnet2_charlesssg data=segmentation/shapenet-fixed

# (failed) You must specify 'models', e.g, models=<OPTION>

# cd ~/Documents/tree_segmentation/torch-points3d/

python torch_points3d/datasets/regression/superquadrics.py


python

from torch_points3d.applications.pretrained_api import PretainedRegistry
model = PretainedRegistry.from_pretrained("pointnet2_largemsg-s3dis-1")



#############################################################
on Burwood
ssh 10.10.37.152
conda activate pyg

sudo dpkg --remove --force-all cuda-repo-ubuntu2004-11-4-local
export PATH="/usr/local/cuda-11.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"

conda install ipykernel
ipython kernel install --user --name=mink # name should be your convda env name!
# conda install -c conda-forge nb_conda_kernels
pip install jupyter
jupyter notebook --no-browser --port=8020

# on local
ssh -N -L 8090:localhost:8090 davidsos@10.10.37.152

# LEEZA
# jupyter notebook --no-browser --port=8888

# ssh -N -L 8888:localhost:8888 leeza@10.10.37.152

## new pyg
cd /home/scion.local/davidsos/Documents
jupyter notebook --no-browser --port=8035
# on local
ssh -N -L 8030:localhost:8030 davidsos@10.10.37.152

# pyg new address (old env) 
ssh -N -L 8035:localhost:8035 davidsos@10.10.37.152

jupyter notebook --no-browser --port=8041
ssh -N -L 8041:localhost:8041 davidsos@10.10.37.152

# for tensorRT Inside docker
ssh -N -L 8888:localhost:8888 davidsos@10.10.37.152

## SMP burwood
conda install ipykernel
ipython kernel install --user --name=smp # name should be your convda env name!
pip install jupyter

## PYG R2D2 2023
conda install ipykernel
ipython kernel install --user --name=smp # name should be your convda env name!
pip install jupyter


# attempted to install latest pyg
ipython kernel install --user --name=pyg_22 # name should be your convda env name!
conda uninstall pyg -c pyg
pip install pyg-lib torch-scatter torch-sparse -f http://data.pyg.org/whl/torch-1.12.0+11.6.html --trusted-host data.pyg.org
pip install torch-geometric

pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

conda install ipykernel
ipython kernel install --user --name=onnxgpu # name should be your convda env name!


# TMUX for ONNXGPU (session), launch Jupyter
jupyter notebook --no-browser --port=8055

# on local
ssh -N -L 8055:localhost:8055davidsos@10.10.37.152


jupyter notebook --no-browser --port=8010

# on local
ssh -N -L 8010:localhost:8010 davidsos@10.10.37.152

# on R2D2
ssh -N -L 8010:localhost:8010 davidsos@10.10.36.163

conda install ipykernel
ipython kernel install --user --name=mmdet # name should be your convda env name!
pip install jupyter

pip install detectron2 -f http://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html --trusted-host dl.fbaipublicfiles.com

conda install ipykernel
ipython kernel install --user --name=detectron2 # name should be your convda env name!
pip install jupyter

conda install ipykernel
ipython kernel install --user --name=mmdeploy # name should be your convda env name!

ipython kernel install --user --name=smp_new # name should be your convda env name!

sudo echo 'export PATH=/home/scion.local/davidsos/anaconda3/bin:$PATH' >> ~/.bashrc
source .bashrc
export PATH=/home/scion.local/davidsos/anaconda3/bin:$PATH


## TensorRT
conda install ipykernel
ipython kernel install --user --name=tensorrt # name should be your convda env name!
# Installed kernelspec tensorrt in /home/davidsos/.local/share/jupyter/kernels/tensorrt
pip install jupyter

jupyter notebook --generate-config
jupyter notebook password

ssh -i "location of your .pem file" -N -f -L 8888:localhost:8888 ubuntu@IP_of_your_remote_server

cd detectron
conda install ipykernel
ipython kernel install --user --name=detectron2 # name should be your convda env name!


tmp/tmp6vi1o652.onnx

python
import tensorrt
print(tensorrt.__version__)
assert tensorrt.Builder(tensorrt.Logger())

https://github.com/pytorch/TensorRT/issues/1026

conda install pytorch torchvision cudatoolkit=11.6 -c pytorch


https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874
(points3d) davidsos@C26789:~$ ipython kernel install --user --name=points3d

# R2D2
ssh 10.10.36.163
another local terminal:
jupyter notebook --no-browser --port=8070

ssh -N -L 8070:localhost:8070 davidsos@10.10.36.163

ps -ef | grep python
pkill python

df -h /dev/shm
monitor shared memory:
watch -n .3 df -h




# Make sure `g++-7 --version` is at least 7.4.0
conda create -n mink python=3.6
conda activate mink

# conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch #latest from torch website
# conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia # from MINK github
# conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch -c conda-forge

# conda install pytorch=1.9.0 torchvision cudatoolkit=11.6 -c pytorch -c nvidia 
conda install openblas-devel -c anaconda

pip3 install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# Looking in links: https://download.pytorch.org/whl/cu113/torch_stable.html

# Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 10.2
export CUDA_HOME=/usr/local/cuda-11.6 #11.6 even though I installed pip 11.3!!!
pip install ninja
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
pip install open3d






conda create -n forest3d python=3.7
pip instsall requirements
pip instsall requirements_seg
pip uninstall tensorflow
conda install -c anaconda cudatoolkit # it chose 11.0.1
pip install tensorflow-gpu==2.4 # 2 == for pip

## WORKS! it's training on R2D2 3090. Seems slow but it's just cause batch size is 3x as large.


docker build -t simtreels -f SimTreeLS_Dockerfile.dockerfile .