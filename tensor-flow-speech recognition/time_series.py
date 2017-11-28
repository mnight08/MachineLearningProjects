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


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
train_audio_path = 'C:/Users/vpx365/Documents/Learning_Data/tensor-flow-word-recognition/train/audio/'


words = ['yes']#[f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]


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


