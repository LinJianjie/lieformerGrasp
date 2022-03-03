add a pytorch_utils
for using the extentions_build --> 

1. install cuda-toolkit depends cuda 10.2 
   https://developer.nvidia.com/cuda-toolkit-archive
   
    1. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    2. sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    3. wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
    4. sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
    5. sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
    6. sudo apt-get update
    7.  sudo apt-get -y install cuda
2. 
    1. export PATH=/usr/local/cuda-10.2/bin${PATH:+:$PATH}}
    2. export LD_LIBRARY_PATH=/usr/local/cuda 10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    
3. Install CUDNN 
    1. tar -xzvf cudnn-10.x-linux-x64-vxxxxx.tgz
    2. sudo cp cuda/include/cudnn.h /usr/local/cuda/include
    3. sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    4. sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

4. conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch