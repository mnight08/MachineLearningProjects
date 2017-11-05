# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 02:12:48 2017

@author: vpx365
"""
from math import sqrt

from joblib import Parallel, delayed



if __name__=="__main__":
    print(Parallel(n_jobs=3)(delayed(sqrt)(i ** 2) for i in range(10)))
