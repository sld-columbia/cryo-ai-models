# Cryo AI Models

## Clone Repository
```
git clone git@github.com:sld-columbia/cryo-ai-models.git
```

## Python Environment Setup
```
conda create --copy --name cryo-ai-env python=3.6
conda activate cryo-ai-env
# pip install git+https://github.com/fastmachinelearning/hls4ml.git#egg=hls4ml[profiling]
pip install git+https://github.com/ruishi31/hls4ml.git@catapult-backend
pip install tensorflow
pip install git+https://github.com/google/qkeras.git#egg=qkeras
vim $HOME/miniconda3/envs/cryo-ai-env/lib/python3.6/site-packages/hls4ml/converters/keras_to_hls.py +223
# You should remove ".decode('utf-8')"
```

## Dataset
```
cd dataset
make download
```

## Getting Started
An existing hls4ml project for the Catapult HLS back-end is maintained in the directory `models/ad03/anomaly_detector_prj`. The following commands will overwrite the project so you can _git diff_ the manual changes that are still necessary in Catapult back-end.

conda activate cryo-ai-env
cd models/ad03
make run-console
# or
make run-python
```
## Compare Vivado HLS vs. Catapult HLS
```
conda activate cryo-ai-env
cd models/ad03
make run-console-vivado
cd anomaly_detector_vivado_prj
vim build_prj.tcl
# set 0 lines 7 to 11
# you want only "csim 1"
vivado_hsl -f build_prj.tcl
vimdiff tb_data/csim_results.log ../anomaly_detector_prj/tb_data/catapult_csim_results.log
```
