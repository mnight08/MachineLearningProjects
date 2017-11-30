# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:40:23 2017

@author: vpx365
"""

import os
from os.path import isdir, isfile, join
from pathlib import Path
import pandas as pd

# Math
import numpy as np
import math
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd

import plotly.offline as py

import plotly.graph_objs as go
import plotly.tools as tls

import re

#voice activity detection
import webrtcvad

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
train_audio_path = 'C:/Users/vpx365/Documents/Learning_Data/tensor-flow-word-recognition/train/audio/'


words = ['yes']#[f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]


'''find the longest list in a data dictionary and pad all the signals with zeros.  Two modes are available.  fft' mode pads to the next power of 2.
'''
def zero_pad(data, mode='fft'):
      padded_data=data
      max_length=0;
      for key in data.keys():
            max_length=max(len(data[key]),max_length)

      if mode =='fft':
            new_length=int(math.pow(2,math.ceil(math.log2(max_length))))

      for key in data.keys():
            zeros_to_append=np.zeros(new_length-len(data[key]),dtype=int)

            padded_data[key]=np.concatenate((padded_data[key],zeros_to_append))
      return padded_data




'''Takes  adata frame whose rows are sampled audio recordings, detects the voice content of each recording then moves the recordings so that the first sample of each signal is the start of voice content. Sample rate must be 8000, 16000, 32000, or 48000 '''
def align_voice_content(dataframe,aggressiveness=1):

      vad = webrtcvad.Vad(aggressiveness)


def load_data(words):
      dirs=[f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f)) and f in words]

      data={}
      speakers=[]
      #load all of the training data into a single dataframe
      for d in dirs:
            print(d)

            directory_path=train_audio_path+d+"/"
            filenames=[filename for filename in os.listdir(directory_path) if       filename.endswith(".wav")]

            print(filenames)
            for filename in filenames:
                  if filename not in speakers:
                        speakers.append(filename)
                  filepath = directory_path+filename
                  print(filepath)
                  sample_rate, samples = wavfile.read(filepath)
                  times=np.linspace(0, sample_rate/len(samples), sample_rate)

                  observation_id=d+"/"+filename+"/"
                  data.update({observation_id+"sample":samples})
      return speakers, times, sample_rate, data

speakers, times, sample_rate, data=load_data(words)


