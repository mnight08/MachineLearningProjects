# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 01:53:26 2017

Responsable for managing recording data.

-loading data
-caching data
-saving data to file


There are two types of raw data:
    -words spoken by a person.
        Each recording of a word is has a unique speaker.

        Each word may have several recordings, and speakers


    -background noises.
        Each type of noise hase one recording



Each speaker may have several recordings





@author: vpx365
"""

class DataManager:
    def __init__(self,words,speakers):
        self.data=None
        self.audio_path = path
        self.words=words
        self.speakers=speakers

        self.samples=None
        self.sample_rates=None
        pass

    def load(words, speakers, mode="raw"):
        pass




    def get_speaker(self,recording_name):
        pass
    def get_word(recording_name):
        pass

    def get_recordings_for_speaker(speaker):
        pass

    def set_words(words):
        self.words=words

    def load_data(self):
        pass


    def custom_fft(self, y, fs):
        T = 1.0 / fs
        N = y.shape[0]
        yf = fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        vals = 2.0/N * np.abs(yf[0:N//2])
        return xf, vals

    '''
    Return a list of paths to recordings of the given words, by the given
    speakers
    '''
    def get_recording_paths(words=None, speakers=None):

        waves = [f for f in os.listdir(join(train_audio_path, direct))
                 if f.endswith('.wav')]



