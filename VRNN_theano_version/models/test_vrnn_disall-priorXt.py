#import ipdb
import numpy as np
import theano
import theano.typed_list as TL
import theano.tensor as T
import datetime
import shutil
import os
import matplotlib.pyplot as plt
plt.switch_backend('PS')
import pickle
import cPickle

from cle.cle.cost import BiGMM, KLGaussianGaussian, GMMdisagMulti
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import LSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    EarlyStopping,
    WeightNorm
)
from cle.cle.train.opt import Adam
from cle.cle.utils import init_tparams, sharedX
from cle.cle.utils.compat import OrderedDict
from cle.cle.utils.op import Gaussian_sample, GMM_sample, GMM_sampleY
from cle.cle.utils.gpu_op import concatenate

from VRNN_theano_version.datasets.dataport import Dataport
from VRNN_theano_version.datasets.dataport_utils import fetch_dataport

#appliances = [ 'kettle','microwave']#
appliances = ['air1', 'furnace1', 'refrigerator1',  'clotheswasher1','drye1','dishwasher1', 'kitchenapp1', 'microwave1']

#[ 'kettle','microwave', 'washing machine', 'dish washer' , 'fridge']#
windows = {2859:("2015-01-01", "2016-01-01")}#, 2:("2013-02-27", "2013-04-27")
#windows = {2859:("2015-01-01", "2016-01-01"),6990:("2015-01-01", "2016-01-01"),7951:("2015-01-01", "2016-01-01"),8292:("2015-01-01",  "2016-01-01"),3413:("2015-01-01", "2016-01-01")}

#2014-2017: 50000
listDates = ['2013-08-26 08:20','2013-11-04 11:37','2014-09-28 11:02','2015-04-11 17:41','2015-10-09 18:02','2016-02-18 20:43']

def main(args):
    
    theano.optimizer='fast_compile'
    theano.config.exception_verbosity='high'
    

    trial = int(args['trial'])
    pkl_name = 'vrnn_gmm_%d' % trial
    channel_name = 'mse'

    data_path = args['data_path']
    save_path = args['save_path']#+'/aggVSdisag_distrib/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    period = int(args['period'])
    n_steps = int(args['n_steps'])
    stride_train = int(args['stride_train'])
    stride_test = n_steps
    typeLoad = int(args['typeLoad'])

    flgMSE = int(args['flgMSE'])
    monitoring_freq = int(args['monitoring_freq'])
    epoch = int(args['epoch'])
    batch_size = int(args['batch_size'])
    x_dim = int(args['x_dim'])
    y_dim = int(args['y_dim'])
    z_dim = int(args['z_dim'])
    rnn_dim = int(args['rnn_dim'])
    k = int(args['num_k']) #a mixture of K Gaussian functions
    lr = float(args['lr'])
    origLR = lr
    debug = int(args['debug'])

    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "saving pkl file '%s'" % pkl_name
    print "to the save path '%s'" % save_path

    q_z_dim = 350
    p_z_dim = 400
    p_x_dim = 450
    x2s_dim = 400
    y2s_dim = 200
    z2s_dim = 350
    target_dim = k# As different appliances are separeted in theta_mu1, theta_mu2, etc... each one is just created from k different Gaussians

    model = Model()
    Xtrain, ytrain, Xval, yval, Xtest, ytest, reader = fetch_dataport(data_path, windows, appliances,numApps=-1, period=period,n_steps= n_steps, 
                                              stride_train = stride_train, stride_test = stride_test,
                                              flgAggSumScaled = 1, flgFilterZeros = 1, typeLoad = typeLoad)

    instancesPlot = {0:[10]}
    #instancesPlot = reader.build_dict_instances_plot(listDates, batch_size, Xval.shape[0])
    
    train_data = Dataport(name='train',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         inputX=Xtrain,
                         labels=ytrain)

    X_mean = train_data.X_mean
    X_std = train_data.X_std

    valid_data = Dataport(name='valid',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         X_mean=X_mean,
                         X_std=X_std,
                         inputX=Xval,
                         labels = yval)

    test_data = Dataport(name='valid',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         X_mean=X_mean,
                         X_std=X_std,
                         inputX=Xtest,
                         labels = ytest)

    init_W = InitCell('rand')
    init_U = InitCell('ortho')
    init_b = InitCell('zeros')
    init_b_sig = InitCell('const', mean=0.6)

    x, mask, y , y_mask = train_data.theano_vars()
    scheduleSamplingMask = T.fvector('schedMask')
    x.name = 'x_original'

    if debug:
        x.tag.test_value = np.zeros((15, batch_size, x_dim), dtype=np.float32)
        temp = np.ones((15, batch_size), dtype=np.float32)
        temp[:, -2:] = 0.
        mask.tag.test_value = temp

    """rnn = LSTM(name='rnn',
               parent=['x_1', 'z_1', 'y_1'],
               parent_dim=[x2s_dim, z2s_dim, y2s_dim],
               nout=rnn_dim,
               unit='tanh',
               init_W=mainloop.model.nodes[0].init_W,
               init_U=mainloop.model.nodes[0].init_U,
               init_b=mainloop.model.nodes[0].init_b)

    x_1 = FullyConnectedLayer(name='x_1',
                              parent=['x_t'],
                              parent_dim=[x_dim],
                              nout=x2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    y_1 = FullyConnectedLayer(name='y_1',
                              parent=['y_t'],
                              parent_dim=[y_dim],
                              nout=y2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    z_1 = FullyConnectedLayer(name='z_1',
                              parent=['z_t'],
                              parent_dim=[z_dim],
                              nout=z2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    phi_1 = FullyConnectedLayer(name='phi_1',
                                parent=['x_1', 's_tm1','y_1'],
                                parent_dim=[x2s_dim, rnn_dim, y2s_dim],
                                nout=q_z_dim,
                                unit='relu',
                                init_W=init_W,
                                init_b=init_b)

    phi_mu = FullyConnectedLayer(name='phi_mu',
                                 parent=['phi_1'],
                                 parent_dim=[q_z_dim],
                                 nout=z_dim,
                                 unit='linear',
                                 init_W=init_W,
                                 init_b=init_b)

    phi_sig = FullyConnectedLayer(name='phi_sig',
                                  parent=['phi_1'],
                                  parent_dim=[q_z_dim],
                                  nout=z_dim,
                                  unit='softplus',
                                  cons=1e-4,
                                  init_W=init_W,
                                  init_b=init_b_sig)

    prior_1 = FullyConnectedLayer(name='prior_1',
                                  parent=['x_1','s_tm1'],
                                  parent_dim=[x2s_dim,rnn_dim],
                                  nout=p_z_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    prior_mu = FullyConnectedLayer(name='prior_mu',
                                   parent=['prior_1'],
                                   parent_dim=[p_z_dim],
                                   nout=z_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    prior_sig = FullyConnectedLayer(name='prior_sig',
                                    parent=['prior_1'],
                                    parent_dim=[p_z_dim],
                                    nout=z_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    theta_1 = FullyConnectedLayer(name='theta_1',
                                  parent=['z_1', 's_tm1'],
                                  parent_dim=[z2s_dim, rnn_dim],
                                  nout=p_x_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    theta_mu1 = FullyConnectedLayer(name='theta_mu1',
                                   parent=['theta_1'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    theta_mu2 = FullyConnectedLayer(name='theta_mu2',
                                   parent=['theta_1'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    theta_mu3 = FullyConnectedLayer(name='theta_mu3',
                                   parent=['theta_1'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    theta_mu4 = FullyConnectedLayer(name='theta_mu4',
                                   parent=['theta_1'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    theta_mu5 = FullyConnectedLayer(name='theta_mu5',
                                   parent=['theta_1'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    theta_sig1 = FullyConnectedLayer(name='theta_sig1',
                                    parent=['theta_1'],
                                    parent_dim=[p_x_dim],
                                    nout=target_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    theta_sig2 = FullyConnectedLayer(name='theta_sig2',
                                    parent=['theta_1'],
                                    parent_dim=[p_x_dim],
                                    nout=target_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    theta_sig3 = FullyConnectedLayer(name='theta_sig3',
                                    parent=['theta_1'],
                                    parent_dim=[p_x_dim],
                                    nout=target_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    theta_sig4 = FullyConnectedLayer(name='theta_sig4',
                                    parent=['theta_1'],
                                    parent_dim=[p_x_dim],
                                    nout=target_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    theta_sig5 = FullyConnectedLayer(name='theta_sig5',
                                    parent=['theta_1'],
                                    parent_dim=[p_x_dim],
                                    nout=target_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    coeff1 = FullyConnectedLayer(name='coeff1',
                                parent=['theta_1'],
                                parent_dim=[p_x_dim],
                                nout=k,
                                unit='softmax',
                                init_W=init_W,
                                init_b=init_b)

    coeff2 = FullyConnectedLayer(name='coeff2',
                                parent=['theta_1'],
                                parent_dim=[p_x_dim],
                                nout=k,
                                unit='softmax',
                                init_W=init_W,
                                init_b=init_b)

    coeff3 = FullyConnectedLayer(name='coeff3',
                                parent=['theta_1'],
                                parent_dim=[p_x_dim],
                                nout=k,
                                unit='softmax',
                                init_W=init_W,
                                init_b=init_b)
 
    coeff4 = FullyConnectedLayer(name='coeff4',
                                parent=['theta_1'],
                                parent_dim=[p_x_dim],
                                nout=k,
                                unit='softmax',
                                init_W=init_W,
                                init_b=init_b)

    coeff5 = FullyConnectedLayer(name='coeff5',
                                parent=['theta_1'],
                                parent_dim=[p_x_dim],
                                nout=k,
                                unit='softmax',
                                init_W=init_W,
                                init_b=init_b)"""

    #from experiment 18-05-31_18-48
    fmodel = open('disall.pkl', 'rb')
    mainloop = cPickle.load(fmodel)
    fmodel.close()

    #define layers
    rnn = mainloop.model.nodes[0]
    x_1 = mainloop.model.nodes[1]
    y_1 = mainloop.model.nodes[2]
    z_1 = mainloop.model.nodes[3]
    phi_1 = mainloop.model.nodes[4]
    phi_mu = mainloop.model.nodes[5]
    phi_sig = mainloop.model.nodes[6]
    prior_1 = mainloop.model.nodes[7]
    prior_mu = mainloop.model.nodes[8]
    prior_sig = mainloop.model.nodes[9]
    theta_1 = mainloop.model.nodes[10]
    theta_mu1 = mainloop.model.nodes[11]
    theta_sig1 = mainloop.model.nodes[12]
    coeff1 = mainloop.model.nodes[13]

    nodes = [rnn,
             x_1, y_1,z_1, #dissag_pred,
             phi_1, phi_mu, phi_sig,
             prior_1, prior_mu, prior_sig,
             theta_1, theta_mu1, theta_sig1, coeff1]

    params = mainloop.model.params

    dynamicOutput = [None, None, None, None, None, None, None, None]
    #dynamicOutput_val = [None, None, None, None, None, None,None,  None, None]
    if (y_dim>1):
      theta_mu2 = mainloop.model.nodes[14]
      theta_sig2 = mainloop.model.nodes[15]
      coeff2 = mainloop.model.nodes[16]
      nodes = nodes + [theta_mu2, theta_sig2, coeff2]
      dynamicOutput = dynamicOutput+[None, None, None, None] #mu, sig, coef and pred
    if (y_dim>2):
      theta_mu3 = mainloop.model.nodes[17]
      theta_sig3 = mainloop.model.nodes[18]
      coeff3 = mainloop.model.nodes[19]
      nodes = nodes + [theta_mu3, theta_sig3, coeff3]
      dynamicOutput = dynamicOutput +[None, None, None, None]
    if (y_dim>3):
      theta_mu4 = mainloop.model.nodes[20]
      theta_sig4 = mainloop.model.nodes[21]
      coeff4 = mainloop.model.nodes[22]
      nodes = nodes + [theta_mu4, theta_sig4, coeff4]
      dynamicOutput = dynamicOutput + [None, None, None, None]
    if (y_dim>4):
      theta_mu5 = mainloop.model.nodes[23]
      theta_sig5 = mainloop.model.nodes[24]
      coeff5 = mainloop.model.nodes[25]
      nodes = nodes + [theta_mu5, theta_sig5, coeff5]
      dynamicOutput = dynamicOutput + [None, None, None, None]

    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)
    y_1_temp = y_1.fprop([y], params)

    output_fn = [s_0] + dynamicOutput
    output_fn_val = [s_0] + dynamicOutput[2:]
    print(len(output_fn), len(output_fn_val))

    def inner_fn_test(x_t, s_tm1):

        prior_1_t = prior_1.fprop([x_t,s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(prior_mu_t, prior_sig_t)#in the original code it is gaussian. GMM is for the generation
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)
        theta_mu1_t = theta_mu1.fprop([theta_1_t], params)
        theta_sig1_t = theta_sig1.fprop([theta_1_t], params)
        coeff1_t = coeff1.fprop([theta_1_t], params)

        y_pred1 = GMM_sampleY(theta_mu1_t, theta_sig1_t, coeff1_t) #Gaussian_sample(theta_mu_t, theta_sig_t)

        tupleMulti = prior_mu_t, prior_sig_t, theta_mu1_t, theta_sig1_t, coeff1_t, y_pred1

        if (y_dim>1):
          theta_mu2_t = theta_mu2.fprop([theta_1_t], params)
          theta_sig2_t = theta_sig2.fprop([theta_1_t], params)
          coeff2_t = coeff2.fprop([theta_1_t], params)
          y_pred2 = GMM_sampleY(theta_mu2_t, theta_sig2_t, coeff2_t)
          y_pred1 = T.concatenate([y_pred1, y_pred2],axis=1)
          tupleMulti = tupleMulti + (theta_mu2_t, theta_sig2_t, coeff2_t, y_pred2)

        if (y_dim>2):
          theta_mu3_t = theta_mu3.fprop([theta_1_t], params)
          theta_sig3_t = theta_sig3.fprop([theta_1_t], params)
          coeff3_t = coeff3.fprop([theta_1_t], params)
          y_pred3 = GMM_sampleY(theta_mu3_t, theta_sig3_t, coeff3_t)
          y_pred1 = T.concatenate([y_pred1, y_pred3],axis=1)
          tupleMulti = tupleMulti + (theta_mu3_t, theta_sig3_t, coeff3_t, y_pred3)

        if (y_dim>3):
          theta_mu4_t = theta_mu4.fprop([theta_1_t], params)
          theta_sig4_t = theta_sig4.fprop([theta_1_t], params)
          coeff4_t = coeff4.fprop([theta_1_t], params)
          y_pred4 = GMM_sampleY(theta_mu4_t, theta_sig4_t, coeff4_t)
          y_pred1 = T.concatenate([y_pred1, y_pred4],axis=1)
          tupleMulti = tupleMulti + (theta_mu4_t, theta_sig4_t, coeff4_t, y_pred4)

        if (y_dim>4):
          theta_mu5_t = theta_mu5.fprop([theta_1_t], params)
          theta_sig5_t = theta_sig5.fprop([theta_1_t], params)
          coeff5_t = coeff5.fprop([theta_1_t], params)
          y_pred5 = GMM_sampleY(theta_mu5_t, theta_sig5_t, coeff5_t)
          y_pred1 = T.concatenate([y_pred1, y_pred5],axis=1)
          tupleMulti = tupleMulti + (theta_mu5_t, theta_sig5_t, coeff5_t, y_pred5)
        
        pred_1_t=y_1.fprop([y_pred1], params)
        #y_pred = [GMM_sampleY(theta_mu_t[i], theta_sig_t[i], coeff_t[i]) for i in range(y_dim)]#T.stack([y_pred1,y_pred2],axis = 0 )
        s_t = rnn.fprop([[x_t, z_1_t, pred_1_t], [s_tm1]], params)
        #y_pred = dissag_pred.fprop([s_t], params)

        return (s_t,)+tupleMulti
        #corr_temp, binary_temp
    (restResults_val, updates_val) = theano.scan(fn=inner_fn_test, sequences=[x_1_temp],
                            outputs_info=output_fn_val )

    for k, v in updates_val.iteritems():
        k.default_update = v

    """def inner_fn(x_t, y_t, s_tm1):

        phi_1_t = phi_1.fprop([x_t, s_tm1, y_t], params)
        phi_mu_t = phi_mu.fprop([phi_1_t], params)
        phi_sig_t = phi_sig.fprop([phi_1_t], params)

        prior_1_t = prior_1.fprop([x_t,s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(phi_mu_t, phi_sig_t)#in the original code it is gaussian. GMM is for the generation
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)
        theta_mu1_t = theta_mu1.fprop([theta_1_t], params)
        theta_sig1_t = theta_sig1.fprop([theta_1_t], params)
        coeff1_t = coeff1.fprop([theta_1_t], params)

        y_pred1 = GMM_sampleY(theta_mu1_t, theta_sig1_t, coeff1_t) #Gaussian_sample(theta_mu_t, theta_sig_t)

        tupleMulti = phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, theta_mu1_t, theta_sig1_t, coeff1_t, y_pred1

        if (y_dim>1):
          theta_mu2_t = theta_mu2.fprop([theta_1_t], params)
          theta_sig2_t = theta_sig2.fprop([theta_1_t], params)
          coeff2_t = coeff2.fprop([theta_1_t], params)
          y_pred2 = GMM_sampleY(theta_mu2_t, theta_sig2_t, coeff2_t)
          tupleMulti = tupleMulti + (theta_mu2_t, theta_sig2_t, coeff2_t, y_pred2)

        if (y_dim>2):
          theta_mu3_t = theta_mu3.fprop([theta_1_t], params)
          theta_sig3_t = theta_sig3.fprop([theta_1_t], params)
          coeff3_t = coeff3.fprop([theta_1_t], params)
          y_pred3 = GMM_sampleY(theta_mu3_t, theta_sig3_t, coeff3_t)
          tupleMulti = tupleMulti + (theta_mu3_t, theta_sig3_t, coeff3_t, y_pred3)

        if (y_dim>3):
          theta_mu4_t = theta_mu4.fprop([theta_1_t], params)
          theta_sig4_t = theta_sig4.fprop([theta_1_t], params)
          coeff4_t = coeff4.fprop([theta_1_t], params)
          y_pred4 = GMM_sampleY(theta_mu4_t, theta_sig4_t, coeff4_t)
          tupleMulti = tupleMulti + (theta_mu4_t, theta_sig4_t, coeff4_t, y_pred4)

        if (y_dim>4):
          theta_mu5_t = theta_mu5.fprop([theta_1_t], params)
          theta_sig5_t = theta_sig5.fprop([theta_1_t], params)
          coeff5_t = coeff5.fprop([theta_1_t], params)
          y_pred5 = GMM_sampleY(theta_mu5_t, theta_sig5_t, coeff5_t)
          tupleMulti = tupleMulti + (theta_mu5_t, theta_sig5_t, coeff5_t, y_pred5)
        
        #y_pred = [GMM_sampleY(theta_mu_t[i], theta_sig_t[i], coeff_t[i]) for i in range(y_dim)]#T.stack([y_pred1,y_pred2],axis = 0 )
        s_t = rnn.fprop([[x_t, z_1_t, y_t], [s_tm1]], params)
        #y_pred = dissag_pred.fprop([s_t], params)

        return (s_t,)+tupleMulti
        #corr_temp, binary_temp
    (restResults, updates) = theano.scan(fn=inner_fn, sequences=[x_1_temp, y_1_temp],
                            outputs_info=output_fn )
    '''
    ((s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp,z_t_temp, z_1_temp, theta_1_temp, 
      theta_mu1_temp, theta_sig1_temp, coeff1_temp, theta_mu2_temp, theta_sig2_temp, coeff2_temp, 
      theta_mu3_temp, theta_sig3_temp, coeff3_temp, theta_mu4_temp, theta_sig4_temp, coeff4_temp,
      theta_mu5_temp, theta_sig5_temp, coeff5_temp, 
      y_pred1_temp, y_pred2_temp, y_pred3_temp, y_pred4_temp, y_pred5_temp), updates) =\
        theano.scan(fn=inner_fn,
                    sequences=[x_1_temp, y_1_temp],
                    outputs_info=[s_0,  None, None, None, None, None, None, None, None,None,  None, None, 
                                  None, None, None, None, None, None, None, None,
                                  None, None, None, None, None, None, None, None])
    '''
    s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp,theta_mu1_temp, theta_sig1_temp, coeff1_temp, y_pred1_temp = restResults[:9]
    restResults = restResults[9:]

    for k, v in updates.iteritems():
        k.default_update = v

    #s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)# seems like this is for creating an additional dimension to s_0


    theta_mu1_temp.name = 'theta_mu1'
    theta_sig1_temp.name = 'theta_sig1'
    coeff1_temp.name = 'coeff1'
    y_pred1_temp.name = 'disaggregation1'

    #[:,:,flgAgg].reshape((y.shape[0],y.shape[1],1)
    mse1 = T.mean((y_pred1_temp - y[:,:,0].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
    mae1 = T.mean( T.abs_(y_pred1_temp - y[:,:,0].reshape((y.shape[0],y.shape[1],1))) )
    mse1.name = 'mse1'
    mae1.name = 'mae1'

    kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)"""

    x_shape = x.shape
    y_shape = y.shape
    x_in = x.reshape((x_shape[0]*x_shape[1], -1))
    y_in = y.reshape((y_shape[0]*y_shape[1], -1))

    """theta_mu1_in = theta_mu1_temp.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig1_in = theta_sig1_temp.reshape((x_shape[0]*x_shape[1], -1))
    coeff1_in = coeff1_temp.reshape((x_shape[0]*x_shape[1], -1))


    ddoutMSEA = []
    ddoutYpreds = [y_pred1_temp]
    
    indexSepDynamic = 6 # plus 1 for TOTAMSE

    
    #totaMSE = T.copy(mse1)
    mse2 = T.zeros((1,))
    mae2 = T.zeros((1,))
    mse3 = T.zeros((1,))
    mae3 = T.zeros((1,))
    mse4 = T.zeros((1,))
    mae4 = T.zeros((1,))
    mse5 = T.zeros((1,))
    mae5 = T.zeros((1,))
    
    if (y_dim>1):
      theta_mu2_temp, theta_sig2_temp, coeff2_temp, y_pred2_temp = restResults[:4]
      restResults = restResults[4:]
      theta_mu2_temp.name = 'theta_mu2'
      theta_sig2_temp.name = 'theta_sig2'
      coeff2_temp.name = 'coeff2'
      y_pred2_temp.name = 'disaggregation2'
      mse2 = T.mean((y_pred2_temp - y[:,:,1].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae2 = T.mean( T.abs_(y_pred2_temp - y[:,:,1].reshape((y.shape[0],y.shape[1],1))) )
      mse2.name = 'mse2'
      mae2.name = 'mae2'

      theta_mu2_in = theta_mu2_temp.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig2_in = theta_sig2_temp.reshape((x_shape[0]*x_shape[1], -1))
      coeff2_in = coeff2_temp.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM = theta_mu2_in, theta_sig2_in, coeff2_in

      ddoutMSEA = ddoutMSEA + [mse2, mae2]
      ddoutYpreds = ddoutYpreds + [y_pred2_temp]
      #totaMSE+=mse2
      indexSepDynamic +=2

    if (y_dim>2):
      theta_mu3_temp, theta_sig3_temp, coeff3_temp, y_pred3_temp = restResults[:4]
      restResults = restResults[4:]
      theta_mu3_temp.name = 'theta_mu3'
      theta_sig3_temp.name = 'theta_sig3'
      coeff3_temp.name = 'coeff3'
      y_pred3_temp.name = 'disaggregation3'
      mse3 = T.mean((y_pred3_temp - y[:,:,2].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae3 = T.mean( T.abs_(y_pred3_temp - y[:,:,2].reshape((y.shape[0],y.shape[1],1))) )
      mse3.name = 'mse3'
      mae3.name = 'mae3'

      theta_mu3_in = theta_mu3_temp.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig3_in = theta_sig3_temp.reshape((x_shape[0]*x_shape[1], -1))
      coeff3_in = coeff3_temp.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM = argsGMM + (theta_mu3_in, theta_sig3_in, coeff3_in)
      ddoutMSEA = ddoutMSEA + [mse3, mae3]
      ddoutYpreds = ddoutYpreds + [y_pred3_temp]
      #totaMSE+=mse3
      indexSepDynamic +=2

    if (y_dim>3):
      theta_mu4_temp, theta_sig4_temp, coeff4_temp, y_pred4_temp = restResults[:4]
      restResults = restResults[4:]
      theta_mu4_temp.name = 'theta_mu4'
      theta_sig4_temp.name = 'theta_sig4'
      coeff4_temp.name = 'coeff4'
      y_pred4_temp.name = 'disaggregation4'
      mse4 = T.mean((y_pred4_temp - y[:,:,3].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae4 = T.mean( T.abs_(y_pred4_temp - y[:,:,3].reshape((y.shape[0],y.shape[1],1))) )
      mse4.name = 'mse4'
      mae4.name = 'mae4'

      theta_mu4_in = theta_mu4_temp.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig4_in = theta_sig4_temp.reshape((x_shape[0]*x_shape[1], -1))
      coeff4_in = coeff4_temp.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM = argsGMM + (theta_mu4_in, theta_sig4_in, coeff4_in)
      ddoutMSEA = ddoutMSEA + [mse4, mae4]
      ddoutYpreds = ddoutYpreds + [y_pred4_temp]
      #totaMSE+=mse4
      indexSepDynamic +=2

    if (y_dim>4):
      theta_mu5_temp, theta_sig5_temp, coeff5_temp, y_pred5_temp = restResults[:4]
      restResults = restResults[4:]
      theta_mu5_temp.name = 'theta_mu5'
      theta_sig5_temp.name = 'theta_sig5'
      coeff5_temp.name = 'coeff5'
      y_pred5_temp.name = 'disaggregation5'
      mse5 = T.mean((y_pred5_temp - y[:,:,4].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae5 = T.mean( T.abs_(y_pred5_temp - y[:,:,4].reshape((y.shape[0],y.shape[1],1))) )
      mse5.name = 'mse5'
      mae5.name = 'mae5'

      theta_mu5_in = theta_mu5_temp.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig5_in = theta_sig5_temp.reshape((x_shape[0]*x_shape[1], -1))
      coeff5_in = coeff5_temp.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM = argsGMM + (theta_mu5_in, theta_sig5_in, coeff5_in)
      ddoutMSEA = ddoutMSEA + [mse5, mae5]
      ddoutYpreds = ddoutYpreds + [y_pred5_temp]
      #totaMSE+=mse5
      indexSepDynamic +=2

    totaMSE = (mse1+mse2+mse3+mse4+mse5)/y_dim
    totaMSE.name = 'mse'

    kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)

    x_shape = x.shape
    y_shape = y.shape
    x_in = x.reshape((x_shape[0]*x_shape[1], -1))
    y_in = y.reshape((y_shape[0]*y_shape[1], -1))
    

    recon = GMMdisagMulti(y_dim, y_in, theta_mu1_in, theta_sig1_in, coeff1_in, *argsGMM)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon = recon.reshape((x_shape[0], x_shape[1]))
    recon.name = 'gmm_out'"""

    '''
    recon5 = GMM(y_in[:,4, None], theta_mu5_in, theta_sig5_in, coeff5_in)
    recon5 = recon.reshape((x_shape[0], x_shape[1]))    
    '''
    """recon_term = recon.sum(axis=0).mean()
    recon_term = recon.sum(axis=0).mean()
    recon_term.name = 'recon_term'

    #kl_temp = kl_temp * mask
    
    kl_term = kl_temp.sum(axis=0).mean()
    kl_term.name = 'kl_term'

    #nll_upper_bound_0 = recon_term + kl_term
    #nll_upper_bound_0.name = 'nll_upper_bound_0'
    if (flgMSE==1):
      nll_upper_bound =  recon_term + kl_term + totaMSE
    else:
      nll_upper_bound =  recon_term + kl_term
    nll_upper_bound.name = 'nll_upper_bound'"""

    ######################## TEST (GENERATION) TIME
    s_temp_val, prior_mu_temp_val, prior_sig_temp_val, \
      theta_mu1_temp_val, theta_sig1_temp_val, coeff1_temp_val, y_pred1_temp_val = restResults_val[:7]
    restResults_val = restResults_val[7:]

    s_temp_val = concatenate([s_0[None, :, :], s_temp_val[:-1]], axis=0)# seems like this is for creating an additional dimension to s_0

    theta_mu1_temp_val.name = 'theta_mu1_val'
    theta_sig1_temp_val.name = 'theta_sig1_val'
    coeff1_temp_val.name = 'coeff1_val'
    y_pred1_temp_val.name = 'disaggregation1_val'

    #[:,:,flgAgg].reshape((y.shape[0],y.shape[1],1)
    mse1_val = T.mean((y_pred1_temp_val - y[:,:,0].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
    mae1_val = T.mean( T.abs_(y_pred1_temp_val - y[:,:,0].reshape((y.shape[0],y.shape[1],1))) )

    #NEURALNILM #(sum_output - sum_target) / max(sum_output, sum_target))
    totPred = T.sum(y_pred1_temp_val)
    totReal = T.sum(y[:,:,0])
    relErr1_val =( totPred -  totReal)/ T.maximum(totPred,totReal)
    propAssigned1_val = 1 - T.sum(T.abs_(y_pred1_temp_val - y[:,:,0].reshape((y.shape[0],y.shape[1],1))))/(2*T.sum(x))

    #y_unNormalize = (y[:,:,0] * reader.stdTraining[0]) + reader.meanTraining[0]
    #y_pred1_temp_val = (y_pred1_temp_val * reader.stdTraining[0]) + reader.meanTraining[0]

    #mse1_valUnNorm = T.mean((y_pred1_temp_val - y_unNormalize.reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
    #mae1_valUnNorm = T.mean( T.abs_(y_pred1_temp_val - y_unNormalize.reshape((y.shape[0],y.shape[1],1))))
    mse1_val.name = 'mse1_val'
    mae1_val.name = 'mae1_val'

    theta_mu1_in_val = theta_mu1_temp_val.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig1_in_val = theta_sig1_temp_val.reshape((x_shape[0]*x_shape[1], -1))
    coeff1_in_val = coeff1_temp_val.reshape((x_shape[0]*x_shape[1], -1))


    ddoutMSEA_val = []
    ddoutYpreds_val = [y_pred1_temp_val]
    totaMSE_val = mse1_val
    totaMAE_val =mae1_val
    indexSepDynamic_val = 5
    prediction_val = y_pred1_temp_val

    #Initializing values of mse and mae
    mse2_val = T.zeros((1,))
    mae2_val = T.zeros((1,))
    mse3_val = T.zeros((1,))
    mae3_val = T.zeros((1,))
    mse4_val = T.zeros((1,))
    mae4_val = T.zeros((1,))
    mse5_val = T.zeros((1,))
    mae5_val = T.zeros((1,))

    relErr2_val = T.zeros((1,))
    relErr3_val = T.zeros((1,))
    relErr4_val = T.zeros((1,))
    relErr5_val = T.zeros((1,))
    propAssigned2_val = T.zeros((1,))
    propAssigned3_val = T.zeros((1,))
    propAssigned4_val = T.zeros((1,))
    propAssigned5_val = T.zeros((1,))
    
    if (y_dim>1):
      theta_mu2_temp_val, theta_sig2_temp_val, coeff2_temp_val, y_pred2_temp_val = restResults_val[:4]
      restResults_val = restResults_val[4:]
      theta_mu2_temp_val.name = 'theta_mu2_val'
      theta_sig2_temp_val.name = 'theta_sig2_val'
      coeff2_temp_val.name = 'coeff2_val'
      y_pred2_temp_val.name = 'disaggregation2_val'
      mse2_val = T.mean((y_pred2_temp_val - y[:,:,1].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae2_val = T.mean( T.abs_(y_pred2_temp_val - y[:,:,1].reshape((y.shape[0],y.shape[1],1))) )

      totPred = T.sum(y_pred2_temp_val)
      totReal = T.sum(y[:,:,1])
      relErr2_val =( totPred -  totReal)/ T.maximum(totPred,totReal)
      propAssigned2_val = 1 - T.sum(T.abs_(y_pred2_temp_val - y[:,:,1].reshape((y.shape[0],y.shape[1],1))))/(2*T.sum(x))

      mse2_val.name = 'mse2_val'
      mae2_val.name = 'mae2_val'

      theta_mu2_in_val = theta_mu2_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig2_in_val = theta_sig2_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      coeff2_in_val = coeff2_temp_val.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM_val = theta_mu2_in_val, theta_sig2_in_val, coeff2_in_val

      ddoutMSEA_val = ddoutMSEA_val + [mse2_val, mae2_val]
      ddoutYpreds_val = ddoutYpreds_val + [y_pred2_temp_val]
      totaMSE_val+=mse2_val
      totaMAE_val+=mae2_val
      indexSepDynamic_val +=2

      prediction_val = T.concatenate([prediction_val, y_pred2_temp_val], axis=2)

    if (y_dim>2):
      theta_mu3_temp_val, theta_sig3_temp_val, coeff3_temp_val, y_pred3_temp_val = restResults_val[:4]
      restResults_val = restResults_val[4:]
      theta_mu3_temp_val.name = 'theta_mu3_val'
      theta_sig3_temp_val.name = 'theta_sig3_val'
      coeff3_temp_val.name = 'coeff3_val'
      y_pred3_temp_val.name = 'disaggregation3_val'
      mse3_val = T.mean((y_pred3_temp_val - y[:,:,2].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae3_val = T.mean( T.abs_(y_pred3_temp_val - y[:,:,2].reshape((y.shape[0],y.shape[1],1))) )

      totPred = T.sum(y_pred3_temp_val)
      totReal = T.sum(y[:,:,2])
      relErr3_val =( totPred -  totReal)/ T.maximum(totPred,totReal)
      propAssigned3_val = 1 - T.sum(T.abs_(y_pred3_temp_val - y[:,:,2].reshape((y.shape[0],y.shape[1],1))))/(2*T.sum(x))

      mse3_val.name = 'mse3_val'
      mae3_val.name = 'mae3_val'

      theta_mu3_in_val = theta_mu3_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig3_in_val = theta_sig3_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      coeff3_in_val = coeff3_temp_val.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM_val = argsGMM_val + (theta_mu3_in_val, theta_sig3_in_val, coeff3_in_val)
      ddoutMSEA_val = ddoutMSEA_val + [mse3_val, mae3_val]
      ddoutYpreds_val = ddoutYpreds_val + [y_pred3_temp_val]
      totaMSE_val+=mse3_val
      totaMAE_val+=mae3_val
      indexSepDynamic_val +=2

      prediction_val = T.concatenate([prediction_val, y_pred3_temp_val], axis=2)

    if (y_dim>3):
      theta_mu4_temp_val, theta_sig4_temp_val, coeff4_temp_val, y_pred4_temp_val = restResults_val[:4]
      restResults_val = restResults_val[4:]
      theta_mu4_temp_val.name = 'theta_mu4_val'
      theta_sig4_temp_val.name = 'theta_sig4_val'
      coeff4_temp_val.name = 'coeff4_val'
      y_pred4_temp_val.name = 'disaggregation4_val'
      mse4_val = T.mean((y_pred4_temp_val - y[:,:,3].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae4_val = T.mean( T.abs_(y_pred4_temp_val - y[:,:,3].reshape((y.shape[0],y.shape[1],1))) )

      totPred = T.sum(y_pred4_temp_val)
      totReal = T.sum(y[:,:,3])
      relErr4_val =( totPred -  totReal)/ T.maximum(totPred,totReal)
      propAssigned4_val = 1 - T.sum(T.abs_(y_pred4_temp_val - y[:,:,3].reshape((y.shape[0],y.shape[1],1))))/(2*T.sum(x))

      mse4_val.name = 'mse4_val'
      mae4_val.name = 'mae4_val'

      theta_mu4_in_val = theta_mu4_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig4_in_val = theta_sig4_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      coeff4_in_val = coeff4_temp_val.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM_val = argsGMM_val + (theta_mu4_in_val, theta_sig4_in_val, coeff4_in_val)
      ddoutMSEA_val = ddoutMSEA_val + [mse4_val, mae4_val]
      ddoutYpreds_val = ddoutYpreds_val + [y_pred4_temp_val]
      totaMSE_val+=mse4_val
      totaMAE_val+=mae4_val
      indexSepDynamic_val +=2
      prediction_val = T.concatenate([prediction_val, y_pred4_temp_val], axis=2)

    if (y_dim>4):
      theta_mu5_temp_val, theta_sig5_temp_val, coeff5_temp_val, y_pred5_temp_val = restResults_val[:4]
      restResults_val = restResults_val[4:]
      theta_mu5_temp_val.name = 'theta_mu5_val'
      theta_sig5_temp_val.name = 'theta_sig5_val'
      coeff5_temp_val.name = 'coeff5_val'
      y_pred5_temp_val.name = 'disaggregation5_val'
      mse5_val = T.mean((y_pred5_temp_val - y[:,:,4].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae5_val = T.mean( T.abs_(y_pred5_temp_val - y[:,:,4].reshape((y.shape[0],y.shape[1],1))))

      totPred = T.sum(y_pred5_temp_val)
      totReal = T.sum(y[:,:,4])
      relErr5_val =( totPred -  totReal)/ T.maximum(totPred,totReal)
      propAssigned5_val = 1 - T.sum(T.abs_(y_pred5_temp_val - y[:,:,4].reshape((y.shape[0],y.shape[1],1))))/(2*T.sum(x))

      mse5_val.name = 'mse5_val'
      mae5_val.name = 'mae5_val'

      theta_mu5_in_val = theta_mu5_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig5_in_val = theta_sig5_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      coeff5_in_val = coeff5_temp_val.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM_val = argsGMM_val + (theta_mu5_in_val, theta_sig5_in_val, coeff5_in_val)
      ddoutMSEA_val = ddoutMSEA_val + [mse5_val, mae5_val]
      ddoutYpreds_val = ddoutYpreds_val + [y_pred5_temp_val]
      totaMSE_val+=mse5_val
      totaMAE_val+=mae5_val
      indexSepDynamic_val +=2
      prediction_val = T.concatenate([prediction_val, y_pred5_temp_val], axis=2)

    recon_val = GMMdisagMulti(y_dim, y_in, theta_mu1_in_val, theta_sig1_in_val, coeff1_in_val, *argsGMM_val)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon_val = recon_val.reshape((x_shape[0], x_shape[1]))
    recon_val.name = 'gmm_out'
    totaMSE_val=totaMSE_val/y_dim
    totaMAE_val=totaMAE_val/y_dim
    
    '''
    recon5 = GMM(y_in[:,4, None], theta_mu5_in, theta_sig5_in, coeff5_in)
    recon5 = recon.reshape((x_shape[0], x_shape[1]))    
    '''
    recon_term_val = recon_val.sum(axis=0).mean()
    recon_term_val = recon_val.sum(axis=0).mean()
    recon_term_val.name = 'recon_term'

    ######################

    model.inputs = [x, mask, y, y_mask, scheduleSamplingMask]
    model.params = params
    model.nodes = nodes

    optimizer = Adam(
        lr=lr
    )
    header = "epoch,log,kl,nll_upper_bound,mse,mae\n"
    extension = [
        GradientClipping(batch_size=batch_size),
        EpochCount(epoch, save_path, header),
        Monitoring(freq=monitoring_freq,
                   #ddout=[nll_upper_bound, recon_term, totaMSE, kl_term, mse1, mae1]+ddoutMSEA+ddoutYpreds ,
                   #indexSep=indexSepDynamic,
                   indexDDoutPlot = [13], # adding indexes of ddout for the plotting
                   #, (6,y_pred_temp)
                   instancesPlot = instancesPlot,#0-150
                   data=[Iterator(valid_data, batch_size)],
                   savedFolder = save_path),
        Picklize(freq=monitoring_freq, path=save_path),
        EarlyStopping(freq=monitoring_freq, path=save_path, channel=channel_name),
        WeightNorm()
    ]

    lr_iterations = {0:lr}

    '''
    mainloop = Training(
        name=pkl_name,
        data=Iterator(train_data, batch_size),
        model=model,
        optimizer=optimizer,
        cost=nll_upper_bound,
        outputs=[nll_upper_bound],
        n_steps = n_steps,
        extension=extension,
        lr_iterations=lr_iterations
    )
    '''

    """mainloop.restore(
        name=pkl_name,
        data=Iterator(train_data, batch_size),
        model=model,
        optimizer=optimizer,
        cost=nll_upper_bound,
        outputs=[nll_upper_bound],
        n_steps = n_steps,
        extension=extension,
        lr_iterations=lr_iterations
    )

    mainloop.run()"""

    
    data=Iterator(test_data, batch_size)

    test_fn = theano.function(inputs=[x, y],#[x, y],
                              #givens={x:Xtest},
                              #on_unused_input='ignore',
                              #z=( ,200,1)
                              allow_input_downcast=True,
                              outputs=[prediction_val, recon_term_val, totaMSE_val, totaMAE_val,
                                        mse1_val,mse2_val,mse3_val,mse4_val,mse5_val,
                                        mae1_val,mae2_val,mae3_val,mae4_val,mae5_val, 
                                        relErr1_val,relErr2_val,relErr3_val,relErr4_val,relErr5_val,
                                        propAssigned1_val, propAssigned2_val,propAssigned3_val,propAssigned4_val,propAssigned5_val]#prediction_val, mse_val, mae_val
                              ,updates=updates_val#, allow_input_downcast=True, on_unused_input='ignore'
                              )
    testOutput = []
    testMetrics2 = []
    numBatchTest = 0
    for batch in data:
      outputGeneration = test_fn(batch[0], batch[2])
      testOutput.append(outputGeneration[1:14])
      testMetrics2.append(outputGeneration[14:])
      #{0:[4,20], 2:[5,10]} 
      #if (numBatchTest==0):

      plt.figure(1)
      plt.plot(np.transpose(outputGeneration[0],[1,0,2])[4])#ORIGINAL 1,0,2
      plt.savefig(save_path+"/vrnn_dis_generated{}_Pred_0-4".format(numBatchTest))
      plt.clf()

      plt.figure(2)
      plt.plot(np.transpose(batch[2],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_RealDisag_0-4".format(numBatchTest))
      plt.clf()

      plt.figure(3)
      plt.plot(np.transpose(batch[0],[1,0,2])[4]) #ORIGINAL 1,0,2
      plt.savefig(save_path+"/vrnn_dis_generated{}_Realagg_0-4".format(numBatchTest))
      plt.clf()
      numBatchTest+=1

    testOutput = np.asarray(testOutput)
    testMetrics2 = np.asarray(testMetrics2)
    print(testOutput.shape)
    print(testMetrics2.shape)
    recon_test =  testOutput[:, 0].mean()
    mse_test =  testOutput[:, 1].mean()
    mae_test =  testOutput[:, 2].mean()
    mse1_test =  testOutput[:, 3].mean()
    mae1_test =  testOutput[:, 8].mean()
    mse2_test =  testOutput[:, 4].mean()
    mae2_test =  testOutput[:, 9].mean()
    mse3_test =  testOutput[:, 5].mean()
    mae3_test =  testOutput[:, 10].mean()
    mse4_test =  testOutput[:, 6].mean()
    mae4_test =  testOutput[:, 11].mean()
    mse5_test =  testOutput[:, 7].mean()
    mae5_test =  testOutput[:, 12].mean()

    relErr1_test = testMetrics2[:,0].mean()
    relErr2_test = testMetrics2[:,1].mean()
    relErr3_test = testMetrics2[:,2].mean()
    relErr4_test = testMetrics2[:,3].mean()
    relErr5_test = testMetrics2[:,4].mean()

    propAssigned1_test = testMetrics2[:, 5].mean()
    propAssigned2_test = testMetrics2[:, 6].mean()
    propAssigned3_test = testMetrics2[:, 7].mean()
    propAssigned4_test = testMetrics2[:, 8].mean()
    propAssigned5_test = testMetrics2[:, 9].mean()

    fLog = open(save_path+'/output.csv', 'w')
    fLog.write(str(lr_iterations)+"\n")
    fLog.write(str(appliances)+"\n")
    fLog.write(str(windows)+"\n")
    fLog.write("logTest,mse1_test,mse2_test,mse3_test,mse4_test,mse5_test,mae1_test,mae2_test,mae3_test,mae4_test,mae5_test,mseTest,maeTest\n")
    fLog.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n\n".format(recon_test,mse1_test,mse2_test,mse3_test, mse4_test,mse5_test,mae1_test,mae2_test,mae3_test, mae4_test,mae5_test,mse_test,mae_test))
    fLog.write("relErr1,relErr2,relErr3,relErr4,relErr5,propAssigned1,propAssigned2,propAssigned3,propAssigned4,propAssigned5\n")
    fLog.write("{},{},{},{},{},{},{},{},{},{}\n".format(relErr1_test,relErr2_test,relErr3_test,relErr4_test, relErr5_test,propAssigned1_test,propAssigned2_test,propAssigned3_test, propAssigned4_test,propAssigned5_test))

    fLog.write("q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim\n")
    fLog.write("{},{},{},{},{},{}\n".format(q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim))
    fLog.write("epoch,log,kl,mse1,mse2,mse3,mse4,mse5,mae1,mae2,mae3,mae4,mae5\n")
    for i , item in enumerate(mainloop.trainlog.monitor['nll_upper_bound']):
      d, e, f,g,j,k,l,m =  0,0,0,0,0,0,0,0
      ep = mainloop.trainlog.monitor['epoch'][i]
      a = mainloop.trainlog.monitor['recon_term'][i]
      b = mainloop.trainlog.monitor['kl_term'][i]
      c = mainloop.trainlog.monitor['mse1'][i]
      h = mainloop.trainlog.monitor['mae1'][i]
      
      if (y_dim>1):
        d = mainloop.trainlog.monitor['mse2'][i]
        j = mainloop.trainlog.monitor['mae2'][i]
      if (y_dim>2):
        e = mainloop.trainlog.monitor['mse3'][i]
        k = mainloop.trainlog.monitor['mae3'][i]
      if (y_dim>3):
        f = mainloop.trainlog.monitor['mse4'][i]
        l = mainloop.trainlog.monitor['mae4'][i]
      if (y_dim>4):
        g = mainloop.trainlog.monitor['mse5'][i]
        m = mainloop.trainlog.monitor['mae5'][i]
      fLog.write("{:d},{:.2f},{:.2f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(ep,a,b,c,d,e,f,g,h,j,k,l,m))

    f = open(save_path+'/outputRealGeneration.pkl', 'wb')
    pickle.dump(outputGeneration, f, -1)
    f.close()

if __name__ == "__main__":

    import sys, time
    if len(sys.argv) > 1:
        config_file_name = sys.argv[-1]
    else:
        config_file_name = 'config_AE-all.txt'

    f = open(config_file_name, 'r')
    lines = f.readlines()
    params = OrderedDict()

    for line in lines:
        line = line.split('\n')[0]
        param_list = line.split(' ')
        param_name = param_list[0]
        param_value = param_list[1]
        params[param_name] = param_value

    params['save_path'] = params['save_path']+'/allAtOnce/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    os.makedirs(params['save_path'])
    shutil.copy('config_AE-all.txt', params['save_path']+'/config_AE-all.txt')

    main(params)