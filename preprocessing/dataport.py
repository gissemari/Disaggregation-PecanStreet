import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import theano.tensor as T

from cle.cle.data import TemporalSeries
from cle.cle.data.prep import SequentialPrepMixin
from cle.cle.utils import segment_axis, tolist, totuple

from dataport_utils import fetch_dataport


class Dataport(TemporalSeries, SequentialPrepMixin): #TemporalSeries comes from class Data (in cle/cle/data/init) and it has ITERATOR
    """
    IAMOnDB dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, prep='none', cond=False, X_mean=None, X_std=None, bias=None, validTime=0, **kwargs):

        self.prep = prep
        self.cond = cond
        self.X_mean = X_mean
        self.X_std = X_std
        self.bias = bias
        self.validTime = validTime
        super(Dataport, self).__init__(**kwargs)

    def load(self, labels, inputX):#load is called when iterating data
        self.labels = [labels]

        return [inputX]

    def theano_vars(self):

        if self.cond:
            return [T.ftensor3('x'), T.fmatrix('mask'),
                    T.ftensor3('y'), T.fmatrix('label_mask')]
        else: # False for training and valid
            return [T.ftensor3('x'), T.fmatrix('mask')]

    def theano_test_vars(self):
        return [T.ftensor3('y'), T.fmatrix('label_mask')]

    def slices(self, start, end):
        
        batches = [mat[start:end] for mat in self.data] #len(self.data[0])=100 I imagine because of [X]
        label_batches = [mat[start:end] for mat in self.labels]
              #mask = self.create_mask(batches[0].swapaxes(0, 1))
        len_batches = len(batches[0].shape) # 1 because (20,)
        if(len_batches <= 1):
            mask = self.create_mask(batches[0])
        else:
            mask = self.create_mask(batches[0].swapaxes(0, 1))
        batches = [self.zero_pad(batch) for batch in batches]
        #label_mask = self.create_mask(label_batches[0].swapaxes(0, 1))
        len_label_batches = len(label_batches[0].shape)
        if(len_label_batches <= 1):
            label_mask = self.create_mask(label_batches[0])
        else:
            label_mask = self.create_mask(label_batches[0].swapaxes(0, 1))
        label_batches = [self.zero_pad(batch) for batch in label_batches]

        if self.cond:
            return totuple([batches[0], mask, label_batches[0], label_mask])
        else:
            return totuple([batches[0], mask])
        '''
        before zero_pad
        (Pdb) len(batches[0])
        20
        (Pdb) len(batches[0][0])
        773
        (Pdb) len(batches[0][1])
        675
        (Pdb) len(batches[0][19])
        280
        (Pdb) len(batches[0][20])
        *** IndexError: index 20
        '''
    def generate_index(self, X):

        maxlen = np.array([len(x) for x in X]).max()
        idx = np.arange(maxlen)

        return idx

if __name__ == "__main__":

    data_path = '/data/lisatmp3/iamondb/'
    iamondb = IAMOnDB(name='train',
                      prep='normalize',
                      cond=False,
                      path=data_path)

    batch = iamondb.slices(start=0, end=10826)
    X = iamondb.data[0]
    sub_X = X

    for item in X:
        max_x = np.max(item[:,1])
        max_y = np.max(item[:,2])
        min_x = np.min(item[:,1])
        min_y = np.min(item[:,2])

    print np.max(max_x)
    print np.max(max_y)
    print np.min(min_x)
    print np.min(min_y)
    ipdb.set_trace()
