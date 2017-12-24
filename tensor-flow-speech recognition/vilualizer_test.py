# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 02:14:41 2017

@author: vpx365
"""

import unittest


from visualization import Visualizer

class VisualizerTest(unittest.TestCase):
    def __init__(self):
        self.all_words=[]
        self.all_speakers=[]

        pass



    '''
    Go through each word and speaker possible, generate the raw figure and
    push to disk for visual inspection. See that the plot functions are called
    successfully.
    '''
    def test_plot_raw(self):
        self.assertEqual('foo'.upper(), 'FOO')

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


if __name__ == '__main__':
    unittest.main()