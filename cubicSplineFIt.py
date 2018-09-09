#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 19:13:19 2018

@author: frederictheunissen
"""

import numpy as np
from soundsig.basis import cubic_spline_basis
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.stats import zscore

x = np.linspace(0, 10, 100)
y = np.exp(-(x-5)**2/2)

x = zscore(x)
y = zscore(y)


y=y+0.1*np.random.randn(y.shape[0])

plt.figure()
plt.plot(x,y)

xsp = cubic_spline_basis(x, num_knots=3)

plt.figure()

for ip in range(6):
    plt.subplot(1,6,ip+1)
    plt.plot(xsp[:,ip], y)

rr = Ridge(alpha=0.01)

rr.fit(xsp, y)

ypred = rr.predict(xsp)

plt.figure()
plt.plot(xsp[:,0], ypred, 'r')
plt.plot(xsp[:,0], y, 'k')
plt.plot(x, y, 'b')
