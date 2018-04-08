from __future__ import division
import os
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import fnmatch
import re
import dataSet_ts as dt
from lxml import etree


def plot_scatter_iamondb_example(X, y=None, equal=True, show=True, save=False,
                                 save_name="tmp.png"):

    rgba_colors = np.zeros((len(X), 4))
    normed = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # for red the first column needs to be one
    rgba_colors[:, 0] = normed[:, 0]

    # for blue last color column needs to be one
    rgba_colors[:, 2] = np.abs(1 - normed[:, 0])

    # the fourth column needs to be alphas
    rgba_colors[:, 3] = np.ones((len(X),)) * .4 + .4 * normed[:, 0]

    if len(X[0]) == 3:
        plt.scatter(X[:, 1], X[:, 2], color=rgba_colors)
    elif len(X[0]) == 2:
        plt.scatter(X[:, 0], X[:, 1], color=rgba_colors)

    if y is not None:
        plt.title(y)

    if equal:
        plt.axis('equal')

    if show:
        if save is True:
            raise ValueError("save cannot be True if show is True!")
        plt.show()
    elif save:
        plt.savefig(save_name)


def plot_lines_iamondb_example(X, y=None, equal=True, show=True, save=False,
                               save_name="tmp.png"):

    val_index = np.where(X[:, 0] != 1)[0]
    contiguous = np.where((val_index[1:] - val_index[:-1]) == 1)[0] + 1
    non_contiguous = np.where((val_index[1:] - val_index[:-1]) != 1)[0] + 1
    prev_nc = 0

    for nc in val_index[non_contiguous]:
        ind = ((prev_nc <= contiguous) & (contiguous < nc))[:-1]
        prev_nc = nc
        plt.plot(X[val_index[ind], 1], X[val_index[ind], 2])
    plt.plot(X[prev_nc:, 1], X[prev_nc:, 2])

    if y is not None:
        plt.title(y)

    if equal:
        plt.axis('equal')

    if show:
        if save is True:
            raise ValueError("save cannot be True if show is True!")
        plt.show()
    elif save:
        plt.savefig(save_name)


def fetch_dataport(data_path, windows, appliances, numApps, period, n_steps, stride_train, stride_test, trainPer=0.5, valPer=0.25, testPer=0.25, typeLoad=0, flgAggSumScaled=0, flgFilterZeros=0):
    '''
    Deleting huge part of seems like generating data from other metadata
    '''
    reader = dt.ReaderTS(windows, appliances, n_steps, stride_train, stride_test, period, flgAggSumScaled, flgFilterZeros,
                        flgScaling=0, trainPer=trainPer, valPer=valPer, testPer=testPer)

    XdataSet, YdataSet = reader.load_csvdata(data_path, numApps,typeLoad)
    #shape before: batch, apps, steps
    x_train, x_test, x_val, y_train, y_test, y_val = XdataSet['train'],XdataSet['test'],XdataSet['val'], YdataSet['train'],YdataSet['test'],YdataSet['val']

    if (numApps==-1):
        return np.expand_dims(x_train,axis=2), y_train, np.expand_dims(x_val,axis=2), y_val, np.expand_dims(x_test,axis=2), y_test, reader  
    return np.expand_dims(x_train,axis=2), np.expand_dims(y_train,axis=2), np.expand_dims(x_val,axis=2), np.expand_dims(y_val,axis=2), np.expand_dims(x_test,axis=2), np.expand_dims(y_test,axis=2), reader
