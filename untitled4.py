#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:33:42 2018

@author: genevieve
"""
import scipy.io

data = scipy.io.loadmat("../Documents/MATLAB/rbm1_vh.mat")
for i in data: 
    print(i)