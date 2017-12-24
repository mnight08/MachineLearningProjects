# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:08:23 2017

@author: vpx365
"""
import os
from os.path import isdir, join
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

from IPython import get_ipython

import timeit

'''Visualizer objects create various visualizations
'''
class Visualizer:

    def __init__(self,words=None):
        self.data=pd.DataFrame()
        self.audio_path = 'C:/Users/vpx365/Documents/Learning_Data/\
                    tensor-flow-word-recognition/train/audio/'
        self.words=[]

        self.figures={}



    def set_words(words):
        self.words=words

    def save_figures():
        pass


    '''generate log_specgram for given audio recording with given window.'''
    def log_specgram(self, audio, sample_rate, window_size=20,
                     step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(audio, fs=sample_rate,
                                                window='hann', nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    '''iterate through the words and count frequency. Then make a bar graph.'''
    def plotWordFrequency(self):
        dirs = [f for f in os.listdir(train_audio_path)
                if isdir(join(train_audio_path, f))]
        dirs.sort()
        print('Number of labels: ' + str(len(dirs)))
        number_of_recordings = []
        for direct in dirs:
            waves = [f for f in os.listdir(join(train_audio_path, direct))
                     if f.endswith('.wav')]
            number_of_recordings.append(len(waves))
            plt.figure(figsize=(14, 8))
            plt.bar(dirs, number_of_recordings)
            plt.title('Number of recordings in given label')
            plt.xticks(rotation='vertical')
            plt.ylabel('Number of recordings')
            plt.xlabel('Words')
            plt.show()

    def custom_fft(self, y, fs):
        T = 1.0 / fs
        N = y.shape[0]
        yf = fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        vals = 2.0/N * np.abs(yf[0:N//2])
        return xf, vals

    '''plots the raw audio recording that it is given'''
    def plot_raw(self, recording, path, samples, sample_rate, save=False,
                 show=False):
        raw_fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(211)
        ax1.set_title('Raw wave of ' + filename)
        ax1.set_ylabel('Amplitude')
        ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate),
                 samples)
        xcoords = [0.025, 0.11, 0.23, 0.49]
        for xc in xcoords:
            ax1.axvline(x=xc*16000, c='r')

        # save the figure if the setting is set.
        if save:
            raw_fig.savefig(recording+"_raw")
        if show:
            raw_fig.show()

    def plot_specgram(self, filename, samples, sample_rate):
        dft_fig = plt.figure(figsize=(14, 8))

        xcoords = [0.025, 0.11, 0.23, 0.49]
        for xc in xcoords:
            ax2.axvline(x=xc, c='r')
            freqs, times, spectrogram_cut = log_specgram(samples_cut,
                                                         sample_rate)
            ax2 = dft_fig.add_subplot(212)
            ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
                       extent=[times.min(), times.max(), freqs.min(),
                               freqs.max()])
            ax2.set_yticks(freqs[::16])
            ax2.set_xticks(times[::16])
            ax2.set_title('Spectrogram of ' + filename)
            ax2.set_ylabel('Freqs in Hz')
            ax2.set_xlabel('Seconds')

    def plot_dft(self, filename, samples, sample_rate):
        xf, vals = custom_fft(samples, sample_rate)
        plt.figure(figsize=(12, 4))
        plt.title('FFT of speaker ' + filename[4:11])
        plt.plot(xf, vals)
        plt.xlabel('Frequency')
        plt.grid()
        plt.figure(figsize=(10, 7))

    '''choose n recordings for each word in the list and compute the mean.
    Then return the plot of that mean.'''
    def plot_mean_raw(self, words, n, save=False, show=False):
        pass

    '''#Pick a sample of size n from the recordings of the given word.
    use that set to find mean, dft.  Then generate the plot.'''
    def plot_mean_dft(self, words, n, save=False, show=False):
        for filename in filenames:
            sample_rate, samples = wavfile.read(str(train_audio_path) +
                                                filename)
            if samples.shape[0] != 16000:
                print(f)
                continue
                xf, vals = custom_fft(samples, 16000)
                vals_all.append(vals)
                freqs, times, spec = log_specgram(samples, 16000)
                spec_all.append(spec)

                plt.figure(figsize=(14, 5))
                plt.subplot(121)
                plt.title('Mean fft of ' + direct)
                plt.plot(np.mean(np.array(vals_all), axis=0))
                plt.grid()
                plt.subplot(122)

    def plot_mean_specgram(self, words, n, save=False, show=False):
        plt.title('Mean specgram of ' + direct)
        plt.imshow(np.mean(np.array(spec_all), axis=0).T, aspect='auto',
                   origin='lower', extent=[times.min(), times.max(),
                                           freqs.min(), freqs.max()])
        plt.yticks(freqs[::16])
        plt.xticks(times[::16])
        plt.show()

    def plot_violin_frequency(self, dirs, freq_ind, save=False, show=False):
        """ Plot violinplots for given words (waves in dirs) and frequency freq_ind
        from all frequencies freqs."""

        spec_all = []  # Contain spectrograms
        ind = 0
        for direct in dirs:
            spec_all.append([])

            waves = [f for f in os.listdir(join(train_audio_path, direct)) if
                     f.endswith('.wav')]
            for wav in waves[:100]:
                sample_rate, samples = wavfile.read(
                    train_audio_path + direct + '/' + wav)
                freqs, times, spec = log_specgram(samples, sample_rate)
                spec_all[ind].extend(spec[:, freq_ind])
            ind += 1

        # Different lengths = different num of frames. Make number equal
        minimum = min([len(spec) for spec in spec_all])
        spec_all = np.array([spec[:minimum] for spec in spec_all])

        plt.figure(figsize=(13, 8))
        plt.title('Frequency ' + str(freqs[freq_ind]) + ' Hz')
        sns.violinplot(data=pd.DataFrame(spec_all.T, columns=dirs))
        if show:
            plt.show()

    '''Returns the following figures for a given recording:
    #-raw plot
    #-DFT plot
    #-Log Specgram
    #assumes that the sample rate is 16000 currently.
    #The figures should be stored in a data frame for efficient access.
    '''
    def visualize_recording(self, filename, samples,
                            sample_rate):
        plot_raw(filename, samples_cut, sample_rate)
        plot_dft(filename, samples, sample_rate)
        plot_specgram(filename, samples, sample_rate)

    '''
    #pick a sample of n recordings(if possible.) of the each of the given words
    #and generates the
    #figures for list of given words:'''
    def visulaize_words(self, word=None, n=1, save=False, show=True):

        filenames = [filename for filename in
                     os.listdir(join(train_audio_path, direct))
                     if f.endswith('.wav') and get_word(filename) == word]
        dirs = [d for d in words if d in to_keep]
        print(words)

        for direct in dirs:
            vals_all = []
            spec_all = []

            waves = [f for f in os.listdir(join(train_audio_path, direct))
                     if f.endswith('.wav')]
            for wav in waves:
                sample_rate, samples = wavfile.read(train_audio_path +
                                                    direct + '/' + wav)

    '''
    create a histogram for the recording lenghth
    '''
    def hist_recording_lengths(self, words):
        num_of_shorter = 0
        for direct in dirs:
            waves = [f for f in os.listdir(join(train_audio_path, direct))
                     if f.endswith('.wav')]
            for wav in waves:
                sample_rate, samples = wavfile.read(train_audio_path +
                                                    direct + '/' + wav)
                if samples.shape[0] < sample_rate:
                    num_of_shorter += 1
                    print('Number of recordings shorter than 1 second: ' +
                          str(num_of_shorter))

    '''
    visualize anomolies
    '''
    def anomoly_detection(self):
        fft_all = []
        names = []
        for direct in dirs:
            waves = [f for f in os.listdir(join(train_audio_path, direct))
                     if f.endswith('.wav')]
            for wav in waves:
                sample_rate, samples = wavfile.read(train_audio_path +
                                                    direct + '/' + wav)
                if samples.shape[0] != sample_rate:
                    samples = np.append(samples, np.zeros(
                            (sample_rate - samples.shape[0], )))
                x, val = custom_fft(samples, sample_rate)
                fft_all.append(val)
                names.append(direct + '/' + wav)
        fft_all = np.array(fft_all)

        # Normalization
        fft_all = (fft_all -
                   np.mean(fft_all, axis=0)) / np.std(fft_all, axis=0)
        # Dim reduction
        pca = PCA(n_components=3)
        fft_all = pca.fit_transform(fft_all)

    def interactive_3d_plot(self, data, names):
        scatt = go.Scatter3d(x=data[:, 0], y=data[:, 1],
                             z=data[:, 2], mode='markers', text=names)
        data = go.Data([scatt])
        layout = go.Layout(title="Anomaly detection")
        figure = go.Figure(data=data, layout=layout)
        py.iplot(figure)

    def create_figs_for_all_words_and_speakers(self, path="./figs"):
        filenames = getnames()

        for word in words:
            pass

    def run(self):
        #start = timeit.default_timer()
        #filename = '/yes/01bb6a2a_nohash_1.wav'

        self.visualize_recording("_yes_0a7c2a8d_nohash_0", samples, sample_rate)



        self.visualize_words(words)
        self.violinplot_frequency(dirs, 20)
        self.violinplot_frequency(dirs, 50)
        self.visualize_recording(filename="_yes_0a7c2a8d_nohash_0")
        self.anomoly_detection()


      #stop = timeit.default_timer()
      #total_time = stop - start

      # output running time in a nice format.
      #mins, secs = divmod(total_time, 60)
      #hours, mins = divmod(mins, 60)

      #print("Total running time: %d:%d:%d.\n"  % (hours, mins, secs))




if __name__ == "__main__":
    visualizer = Visualizer(words = 'yes no up down left right on off \
                stop go silence unknown'.split())
    visualizer.run()
