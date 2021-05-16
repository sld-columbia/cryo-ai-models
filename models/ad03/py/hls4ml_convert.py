import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects

import keras_model

import hls4ml

import yaml

import plotting

def yaml_load(config):
    with open(config, 'r') as stream:
        param = yaml.safe_load(stream)
    return param


model_file = 'keras_model.h5'

model = keras_model.load_model(model_file)

model.summary()

hls_config = yaml_load('hls4ml_config.yml')
hls_config = hls_config['HLSConfig']

backend_config = hls4ml.converters.create_vivado_config(fpga_part='xczu7ev-ffvc1156-2-e')
backend_config['ProjectName'] = 'anomaly_detector'
backend_config['KerasModel'] = model
backend_config['HLSConfig'] = hls_config
backend_config['OutputDir'] = 'anomaly_detector_prj'
backend_config['Backend'] = 'Vivado'
backend_config['Implementation'] = 'serial'
backend_config['ClockPeriod'] = 10

plotting.print_dict(backend_config)

hls_model = hls4ml.converters.keras_to_hls(backend_config)

_ = hls_model.compile()


