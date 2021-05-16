# Cryo AI Models

## Setup
```
conda create --copy --name cryo-ai-env python=3.6
conda activate cryo-ai-env
# pip install git+https://github.com/fastmachinelearning/hls4ml.git#egg=hls4ml[profiling]
pip install git+https://github.com/ruishi31/hls4ml.git@catapult-backend
pip install tensorflow
pip install git+https://github.com/google/qkeras.git#egg=qkeras
vim $HOME/miniconda3/envs/hls4ml-env/lib/python3.6/site-packages/hls4ml/converters/keras_to_hls.py +223
# You should remove ".decode('utf-8')"
```
