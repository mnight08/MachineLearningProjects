# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:14:17 2018
This file will take a data frame and create various visualizations of the 
variables and combinations.
@author: vpx365
"""

from datamanager import DataManager
import matplotlib as plt
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import functools

class DataExplorer(): 
    '''
    Create various plots of the signals, and signal features.

    '''
    def __init__(self, dm):
        self.dm=dm



    
    def plot_spectrogram(self, ids):
        ''''''
        f, t, Sxx = signal.spectrogram(self.dm.train.iloc[:,ids], 800000/20*1000, mode ='magnitude')
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    def plot_periodogram(self, ids):
        f, Pxx_den = signal.periodogram(self.dm.train.iloc[:,ids], 800000/20*1000)
    
    def plot_signal(self, ids):
         x = np.linspace(0,1/50,800000)       
         if isinstance(ids, int):
             ids=[ids]
         #Check if 
         self.dm.load_missing_signals(ids)
         state = self.dm.train_meta.loc[self.dm.train_meta['signal_id'].isin(ids),'target'].sum()
         plt.plot(x,self.dm.train.loc[:,[str(id) for id in ids]])         
         plt.ylabel('Voltage')
         plt.xlabel('Time [sec]')
         plt.title("Signals: "+str(ids)+ " Num Failures: "+ str(state))
         plt.show()
         
        
    def get_index_signals(self,state=0):
        '''Reutrn the columns of signals that are experiencing the state: 0 for good, 1 for partial discharge.'''
        return [str(id) for id in self.dm.train_meta.groupby('target').groups[state]]
    
    def get_index_triple(self,state=0):
        ''''Return the list of integer ids of triples that are experiencing the given state.  0, for good,
        1 for one bad line, 2, 3, etc.'''
        return [id for id in self.dm.train_meta.groupby('target').groups[state]]
    def plot_triple(self, ids):
        if isinstance(ids, int):
            ids=[ids]
        for id in ids:
            self.plot_signal([3*id,3*id+1,3*id+2])
        
    def get_power(self, ids):
        '''return the root mean square error '''
        pass
    
    def plot_statisitc(self):
        pass
    
    def shift_triple(self):
        '''Take a collection of triples, to shift so that phase 0 starts at 0 and rises, phase 1, is delayed by 2pi/3, and phase 2 is delayed by 4pi/3'''
        
        #find the offset of the first signal with its rising from zero part of cycle.
        #Original noise free signal is of the form asin(100pit), measurements are shifted versions. 
        #s1(t)=asin(100pit-shift)
        #s2(t)=asin(100pi(t-2/3)-shift)
        #s3(t)=asin(100pi(t-4/3)-shift)
        #Need to find shift, then shift each signal to the left by that amount. 
        #Let F(s1)=S1(fourier transform), and S1=F(asin(100pit))=aF(sin(100pit)).  
        #S1=F(asin(100pit-shift))=aF(sin(100opit))e^-2piaf=
        #find the distance between the 
        #correlate with base wave: sin(100pi t)
        pass
    
    def get_proportion_of_classes(self):
        '''Find the proportion of signals that experiences partial discharge, and those that did not.'''
        total=self.dm.train_meta.shape[0]
        partial=self.dm.train_meta[self.dm.train_meta['target']==0].shape[1]
        perfect=self.dm.train_meta[self.dm.train_meta['target']==0].shape[0]
        return partial/total, perfect/total
 

    def get_prop_lines_partial(self, num_pdis=0):
        '''Find the proportion of line measurements(three signals) that have num_p phases 
        equal to the given num_pdis.  num_pdis refers to the number of phases that 
        are experiencing partial discharge. 0 for none, ..., 3 for all three lines'''
        
        N = max(dm.train_meta['id_measurement'])
        count=0
        for id in range(0, N+1):
            p1=self.dm.train_meta.loc[3*id,'target']
            p2=self.dm.train_meta.loc[3*id+1,'target']
            p3=self.dm.train_meta.loc[3*id+2,'target']
            #count the number of phases experiencing partial discharge, 
            #and see if it is equal to state.
            if p1+p2+p3 == num_pdis:
                count=count+1
        return count/N
            
            

    def visualize_colesium_results(self, coliseum_results, stages=['Imputer', 'Model'], save=False):
        targets=['test_score', 'fit_time', 'score_time']
        for target in targets:
            for stage in stages:
                ax=self.make_colesium_results_box_plot(coliseum_results, stage, target)
                if save:
                    fig=ax.get_figure()
                    fig.savefig(self.make_path+stage+"-"+target+".png")

    '''
    This will create a box plot for test score for the given colesium results
    '''
    def make_colesium_results_box_plot(self, coliseum_results, stage='Model', target= "test_score"):
       
        return coliseum_results.boxplot(column=target, figsize=(6,6), by=stage)        
        

    

    