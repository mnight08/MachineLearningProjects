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

from figure_manager import FigureManager
from data_manager import DataManager
'''
Responsible for create various figures representing visualizing data.
The figures can be cached in a figure manager.

Visualizer objects are responsible for generating the figures.

Other structures are used to cache the figures and interact with
the data itself.

'''
class Visualizer:

    def __init__(self,path=None, words=None):
        self.data=DataManager()


        # Generated figures will be cached here.
        # The figures will be ided by a key.
        self.figures=FigureManager()




    def cache_figure(self, figid,fig):
        pass

    def save_figure(self, figid,fig):
        pass

    def generate_figures(self,mode=None):
        pass


    #def get_figure_id()

    '''generate log_specgram for given audio recording with given window.'''
    def log_specgram(self, recording, window_size=20,
                     step_size=10, eps=1e-10):
        sample_rate=self.data.get_sample_rate(recording)
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(recording, fs=sample_rate,
                                                window='hann', nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)





    '''
    iterate through the words and count frequency. Then make a bar graph.
    The figure is placed in the figures dictionary.  The key is returned.
    '''
    def plot_word_bar_graph(self, words=None, show=True):

        #if self.words == None and words == None:
        #    {}

        #dirs = [f for f in os.listdir(train_audio_path)
        #       if isdir(join(train_audio_path, f))]
        words.sort()
        #print('Number of labels: ' + str(len(dirs)))

        number_of_recordings = []

        for word in words:
            number_of_recordings.append(len(data.get_recording_paths(word)))

        fig=plt.figure(figsize=(14, 8))
        fig.bar(words, number_of_recordings)
        fig.title('Number of recordings in given label')
        fig.xticks(rotation='vertical')
        fig.ylabel('Number of recordings')
        fig.xlabel('Words')

        if show==True:
            fig.show()
        self.figures.add(fig)
        return fig


    '''plots the raw audio recording that it is given'''
    def plot_raw(self, recordings, save=False,
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

    def plot_specgram(self, recordings):
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

    def plot_dft(self, recording):
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

    '''
    Returns the following figures for a given recording:
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





if __name__ == "__main__":
    visualizer = Visualizer(words = 'yes no up down left right on off \
                stop go silence unknown'.split(),path='C:/Users/vpx365/\
                Documents/Learning_Data/tensor-flow-word-recognition/\
                train/audio/')
    visualizer.run()
