import ipdb
import numpy as np
import theano
import theano.typed_list as TL
import theano.tensor as T
import datetime
import shutil
import os

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

appliances = ['furnace1','refrigerator1']
#[ 'air1', 'furnace1','refrigerator1', 'clotheswasher1','drye1','dishwasher1', 'kitchenapp1','microwave1']
windows = {2859:("2015-01-01", "2015-12-01")}#3413:("2015-06-01", "2015-12-31")
#windows = {6990:("2015-06-01", "2015-11-01"), 2859:("2015-06-01", "2015-11-01"), 7951:("2015-06-01", "2015-11-01"),8292:("2015-06-01",  "2015-11-01"),3413:("2015-06-01", "2015-11-01")}#3413:("2015-06-01", "2015-12-31")

def main(args):
    
    theano.optimizer='fast_compile'
    #theano.config.exception_verbosity='high'
    

    trial = int(args['trial'])
    pkl_name = 'vrnn_gmm_%d' % trial
    channel_name = 'nll_upper_bound'

    data_path = args['data_path']
    save_path = args['save_path']#+'/aggVSdisag_distrib/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    period = int(args['period'])
    n_steps = int(args['n_steps'])
    stride_train = int(args['stride_train'])
    stride_test = int(args['stride_test'])
    loadType = int(args['loadType'])

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

    q_z_dim = 500
    p_z_dim = 500
    p_x_dim = 500
    x2s_dim = 200
    y2s_dim = 200
    z2s_dim = 100
    target_dim = k# As different appliances are separeted in theta_mu1, theta_mu2, etc... each one is just created from k different Gaussians

    model = Model()
    Xtrain, ytrain, Xval, yval, Xtest,ytest, reader = fetch_dataport(data_path, windows, appliances,numApps=-1, period=period,
                                              n_steps= n_steps, stride_train = stride_train, stride_test = stride_test,
                                              trainPer=0.6, valPer=0.2, testPer=0.2, loadType = loadType,
                                              flgAggSumScaled = 1, flgFilterZeros = 1)

    instancesPlot = {0:[4,20], 2:[5,10]}

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

    init_W = InitCell('rand')
    init_U = InitCell('ortho')
    init_b = InitCell('zeros')
    init_b_sig = InitCell('const', mean=0.6)

    x, mask, y , y_mask = train_data.theano_vars()
    x.name = 'x_original'

    if debug:
        x.tag.test_value = np.zeros((15, batch_size, x_dim), dtype=np.float32)
        temp = np.ones((15, batch_size), dtype=np.float32)
        temp[:, -2:] = 0.
        mask.tag.test_value = temp

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

    rnn = LSTM(name='rnn',
               parent=['x_1', 'z_1', 'y_1'],
               parent_dim=[x2s_dim, z2s_dim, y2s_dim],
               nout=rnn_dim,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

    phi_1 = FullyConnectedLayer(name='phi_1',
                                parent=['x_1', 's_tm1'],
                                parent_dim=[x2s_dim, rnn_dim],
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
                                  parent=['s_tm1'],
                                  parent_dim=[rnn_dim],
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

    if (y_dim>1):
      theta_mu2 = FullyConnectedLayer(name='theta_mu2',
                                     parent=['theta_1'],
                                     parent_dim=[p_x_dim],
                                     nout=target_dim,
                                     unit='linear',
                                     init_W=init_W,
                                     init_b=init_b)

    if (y_dim>2):
      theta_mu3 = FullyConnectedLayer(name='theta_mu3',
                                     parent=['theta_1'],
                                     parent_dim=[p_x_dim],
                                     nout=target_dim,
                                     unit='linear',
                                     init_W=init_W,
                                     init_b=init_b)

    if (y_dim>3):
      theta_mu4 = FullyConnectedLayer(name='theta_mu4',
                                     parent=['theta_1'],
                                     parent_dim=[p_x_dim],
                                     nout=target_dim,
                                     unit='linear',
                                     init_W=init_W,
                                     init_b=init_b)

    if (y_dim>4):
      theta_mu5 = FullyConnectedLayer(name='theta_mu5',
                                     parent=['theta_1'],
                                     parent_dim=[p_x_dim],
                                     nout=target_dim,
                                     unit='linear',
                                     init_W=init_W,
                                     init_b=init_b)

    if (y_dim>5):
      theta_mu6 = FullyConnectedLayer(name='theta_mu6',
                                     parent=['theta_1'],
                                     parent_dim=[p_x_dim],
                                     nout=target_dim,
                                     unit='linear',
                                     init_W=init_W,
                                     init_b=init_b)

    if (y_dim>6):
      theta_mu7 = FullyConnectedLayer(name='theta_mu7',
                                     parent=['theta_1'],
                                     parent_dim=[p_x_dim],
                                     nout=target_dim,
                                     unit='linear',
                                     init_W=init_W,
                                     init_b=init_b)

    if (y_dim>7):
      theta_mu8 = FullyConnectedLayer(name='theta_mu8',
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

    if (y_dim>1):
      theta_sig2 = FullyConnectedLayer(name='theta_sig2',
                                      parent=['theta_1'],
                                      parent_dim=[p_x_dim],
                                      nout=target_dim,
                                      unit='softplus',
                                      cons=1e-4,
                                      init_W=init_W,
                                      init_b=init_b_sig)

    if (y_dim>2):
      theta_sig3 = FullyConnectedLayer(name='theta_sig3',
                                      parent=['theta_1'],
                                      parent_dim=[p_x_dim],
                                      nout=target_dim,
                                      unit='softplus',
                                      cons=1e-4,
                                      init_W=init_W,
                                      init_b=init_b_sig)

    if (y_dim>3):
      theta_sig4 = FullyConnectedLayer(name='theta_sig4',
                                      parent=['theta_1'],
                                      parent_dim=[p_x_dim],
                                      nout=target_dim,
                                      unit='softplus',
                                      cons=1e-4,
                                      init_W=init_W,
                                      init_b=init_b_sig)

    if (y_dim>4):
      theta_sig5 = FullyConnectedLayer(name='theta_sig5',
                                      parent=['theta_1'],
                                      parent_dim=[p_x_dim],
                                      nout=target_dim,
                                      unit='softplus',
                                      cons=1e-4,
                                      init_W=init_W,
                                      init_b=init_b_sig)

    if (y_dim>5):
      theta_sig6 = FullyConnectedLayer(name='theta_sig6',
                                      parent=['theta_1'],
                                      parent_dim=[p_x_dim],
                                      nout=target_dim,
                                      unit='softplus',
                                      cons=1e-4,
                                      init_W=init_W,
                                      init_b=init_b_sig)

    if (y_dim>6):
      theta_sig7 = FullyConnectedLayer(name='theta_sig7',
                                      parent=['theta_1'],
                                      parent_dim=[p_x_dim],
                                      nout=target_dim,
                                      unit='softplus',
                                      cons=1e-4,
                                      init_W=init_W,
                                      init_b=init_b_sig)

    if (y_dim>7):
      theta_sig8 = FullyConnectedLayer(name='theta_sig8',
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

    if (y_dim>1):
      coeff2 = FullyConnectedLayer(name='coeff2',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=k,
                                  unit='softmax',
                                  init_W=init_W,
                                  init_b=init_b)

    if (y_dim>2):
      coeff3 = FullyConnectedLayer(name='coeff3',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=k,
                                  unit='softmax',
                                  init_W=init_W,
                                  init_b=init_b)

    if (y_dim>3): 
      coeff4 = FullyConnectedLayer(name='coeff4',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=k,
                                  unit='softmax',
                                  init_W=init_W,
                                  init_b=init_b)

    if (y_dim>4):
      coeff5 = FullyConnectedLayer(name='coeff5',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=k,
                                  unit='softmax',
                                  init_W=init_W,
                                  init_b=init_b)

    if (y_dim>5):
      coeff6 = FullyConnectedLayer(name='coeff6',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=k,
                                  unit='softmax',
                                  init_W=init_W,
                                  init_b=init_b)

    if (y_dim>6):
      coeff7 = FullyConnectedLayer(name='coeff7',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=k,
                                  unit='softmax',
                                  init_W=init_W,
                                  init_b=init_b)

    if (y_dim>7):
      coeff8 = FullyConnectedLayer(name='coeff8',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=k,
                                  unit='softmax',
                                  init_W=init_W,
                                  init_b=init_b)

    corr = FullyConnectedLayer(name='corr',
                               parent=['theta_1'],
                               parent_dim=[p_x_dim],
                               nout=k,
                               unit='tanh',
                               init_W=init_W,
                               init_b=init_b)

    binary = FullyConnectedLayer(name='binary',
                                 parent=['theta_1'],
                                 parent_dim=[p_x_dim],
                                 nout=1,
                                 unit='sigmoid',
                                 init_W=init_W,
                                 init_b=init_b)

    nodes = [rnn,
             x_1, y_1,z_1, #dissag_pred,
             phi_1, phi_mu, phi_sig,
             prior_1, prior_mu, prior_sig,
             theta_1, theta_mu1, theta_sig1, coeff1]

    dynamicOutput = [None, None, None, None, None, None, None, None,None,  None, None]
    if (y_dim>1):
      nodes = nodes + [theta_mu2, theta_sig2, coeff2]
      dynamicOutput = dynamicOutput+[None, None, None, None] #mu, sig, coef and pred
    if (y_dim>2):
      nodes = nodes + [theta_mu3, theta_sig3, coeff3]
      dynamicOutput = dynamicOutput +[None, None, None, None]
    if (y_dim>3):
      nodes = nodes + [theta_mu4, theta_sig4, coeff4]
      dynamicOutput = dynamicOutput + [None, None, None, None]
    if (y_dim>4):
      nodes = nodes + [theta_mu5, theta_sig5, coeff5]
      dynamicOutput = dynamicOutput + [None, None, None, None]
    if (y_dim>5):
      nodes = nodes + [theta_mu6, theta_sig6, coeff6]
      dynamicOutput = dynamicOutput + [None, None, None, None]
    if (y_dim>6):
      nodes = nodes + [theta_mu7, theta_sig7, coeff7]
      dynamicOutput = dynamicOutput + [None, None, None, None]
    if (y_dim>7):
      nodes = nodes + [theta_mu8, theta_sig8, coeff8]
      dynamicOutput = dynamicOutput + [None, None, None, None]

    params = OrderedDict()

    for node in nodes:
        if node.initialize() is not None:
            params.update(node.initialize())

    params = init_tparams(params)

    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)
    y_1_temp = y_1.fprop([y], params)

    output_fn = [s_0] + dynamicOutput
    print(len(output_fn))

    def inner_fn(x_t, y_t, s_tm1):

        phi_1_t = phi_1.fprop([x_t, s_tm1], params)
        phi_mu_t = phi_mu.fprop([phi_1_t], params)
        phi_sig_t = phi_sig.fprop([phi_1_t], params)

        prior_1_t = prior_1.fprop([s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(phi_mu_t, phi_sig_t)#in the original code it is gaussian. GMM is for the generation
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)

        
        theta_mu1_t = theta_mu1.fprop([theta_1_t], params)
        theta_sig1_t = theta_sig1.fprop([theta_1_t], params)
        coeff1_t = coeff1.fprop([theta_1_t], params)

        y_pred1 = GMM_sampleY(theta_mu1_t, theta_sig1_t, coeff1_t) #Gaussian_sample(theta_mu_t, theta_sig_t)

        tupleMulti = phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, z_t,  z_1_t, theta_1_t, theta_mu1_t, theta_sig1_t, coeff1_t, y_pred1

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

        if (y_dim>5):
          theta_mu6_t = theta_mu6.fprop([theta_1_t], params)
          theta_sig6_t = theta_sig6.fprop([theta_1_t], params)
          coeff6_t = coeff6.fprop([theta_1_t], params)
          y_pred6 = GMM_sampleY(theta_mu6_t, theta_sig6_t, coeff6_t)
          tupleMulti = tupleMulti + (theta_mu6_t, theta_sig6_t, coeff6_t, y_pred6)

        if (y_dim>6):
          theta_mu7_t = theta_mu7.fprop([theta_1_t], params)
          theta_sig7_t = theta_sig7.fprop([theta_1_t], params)
          coeff7_t = coeff7.fprop([theta_1_t], params)
          y_pred7 = GMM_sampleY(theta_mu7_t, theta_sig7_t, coeff7_t)
          tupleMulti = tupleMulti + (theta_mu7_t, theta_sig7_t, coeff7_t, y_pred7)

        if (y_dim>7):
          theta_mu8_t = theta_mu8.fprop([theta_1_t], params)
          theta_sig8_t = theta_sig8.fprop([theta_1_t], params)
          coeff8_t = coeff8.fprop([theta_1_t], params)
          y_pred8 = GMM_sampleY(theta_mu8_t, theta_sig8_t, coeff8_t)
          tupleMulti = tupleMulti + (theta_mu8_t, theta_sig8_t, coeff8_t, y_pred8)

        s_t = rnn.fprop([[x_t, z_1_t, y_t], [s_tm1]], params)

        return (s_t,)+tupleMulti

        #corr_temp, binary_temp
    (otherResults, updates) = theano.scan(fn=inner_fn, sequences=[x_1_temp, y_1_temp],
                            outputs_info=output_fn )#[s_0, (None)]

    s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp,z_t_temp, z_1_temp, theta_1_temp, \
      theta_mu1_temp, theta_sig1_temp, coeff1_temp, y_pred1_temp = otherResults[:12]
    restResults = otherResults[12:]

    for k, v in updates.iteritems():
        k.default_update = v

    s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)# seems like this is for creating an additional dimension to s_0

    s_temp.name = 'h_1'#gisse
    z_1_temp.name = 'z_1'#gisse
    z_t_temp.name = 'z'

    theta_mu1_temp.name = 'theta_mu1'
    theta_sig1_temp.name = 'theta_sig1'
    coeff1_temp.name = 'coeff1'
    y_pred1_temp.name = 'disaggregation1'

    #[:,:,flgAgg].reshape((y.shape[0],y.shape[1],1)
    mse1 = T.mean((y_pred1_temp - y[:,:,0].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
    mae1 = T.mean( T.abs_(y_pred1_temp - y[:,:,0].reshape((y.shape[0],y.shape[1],1))) )
    mse1.name = 'mse1'
    mae1.name = 'mae1'

    kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)

    x_shape = x.shape
    y_shape = y.shape
    x_in = x.reshape((x_shape[0]*x_shape[1], -1))
    y_in = y.reshape((y_shape[0]*y_shape[1], -1))

    theta_mu1_in = theta_mu1_temp.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig1_in = theta_sig1_temp.reshape((x_shape[0]*x_shape[1], -1))
    coeff1_in = coeff1_temp.reshape((x_shape[0]*x_shape[1], -1))


    ddoutMSEA = []
    ddoutYpreds = [y_pred1_temp]
    totaMSE = mse1
    indexSepDynamic = 5

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
      totaMSE+=mse2
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
      totaMSE+=mse3
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
      totaMSE+=mse4
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
      totaMSE+=mse5
      indexSepDynamic +=2

    if (y_dim>5):
      theta_mu6_temp, theta_sig6_temp, coeff6_temp, y_pred6_temp = restResults[:4]
      restResults = restResults[4:]
      theta_mu6_temp.name = 'theta_mu6'
      theta_sig6_temp.name = 'theta_sig6'
      coeff6_temp.name = 'coeff6'
      y_pred6_temp.name = 'disaggregation6'
      mse6 = T.mean((y_pred6_temp - y[:,:,2].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae6 = T.mean( T.abs_(y_pred6_temp - y[:,:,2].reshape((y.shape[0],y.shape[1],1))) )
      mse6.name = 'mse6'
      mae6.name = 'mae6'

      theta_mu6_in = theta_mu6_temp.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig6_in = theta_sig6_temp.reshape((x_shape[0]*x_shape[1], -1))
      coeff6_in = coeff6_temp.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM = argsGMM + (theta_mu6_in, theta_sig6_in, coeff6_in)
      ddoutMSEA = ddoutMSEA + [mse6, mae6]
      ddoutYpreds = ddoutYpreds + [y_pred6_temp]
      totaMSE+=mse6
      indexSepDynamic +=2

    if (y_dim>6):
      theta_mu7_temp, theta_sig7_temp, coeff7_temp, y_pred7_temp = restResults[:4]
      restResults = restResults[4:]
      theta_mu7_temp.name = 'theta_mu7'
      theta_sig7_temp.name = 'theta_sig7'
      coeff7_temp.name = 'coeff7'
      y_pred7_temp.name = 'disaggregation7'
      mse7 = T.mean((y_pred7_temp - y[:,:,3].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae7 = T.mean( T.abs_(y_pred7_temp - y[:,:,3].reshape((y.shape[0],y.shape[1],1))) )
      mse7.name = 'mse7'
      mae7.name = 'mae7'

      theta_mu7_in = theta_mu7_temp.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig7_in = theta_sig7_temp.reshape((x_shape[0]*x_shape[1], -1))
      coeff7_in = coeff7_temp.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM = argsGMM + (theta_mu7_in, theta_sig7_in, coeff7_in)
      ddoutMSEA = ddoutMSEA + [mse7, mae7]
      ddoutYpreds = ddoutYpreds + [y_pred7_temp]
      totaMSE+=mse7
      indexSepDynamic +=2

    if (y_dim>7):
      theta_mu8_temp, theta_sig8_temp, coeff8_temp, y_pred8_temp = restResults[:4]
      restResults = restResults[4:]
      theta_mu8_temp.name = 'theta_mu8'
      theta_sig8_temp.name = 'theta_sig8'
      coeff8_temp.name = 'coeff8'
      y_pred8_temp.name = 'disaggregation8'
      mse8 = T.mean((y_pred8_temp - y[:,:,4].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae8 = T.mean( T.abs_(y_pred8_temp - y[:,:,4].reshape((y.shape[0],y.shape[1],1))) )
      mse8.name = 'mse8'
      mae8.name = 'mae8'

      theta_mu8_in = theta_mu8_temp.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig8_in = theta_sig8_temp.reshape((x_shape[0]*x_shape[1], -1))
      coeff8_in = coeff8_temp.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM = argsGMM + (theta_mu8_in, theta_sig8_in, coeff8_in)
      ddoutMSEA = ddoutMSEA + [mse8, mae8]
      ddoutYpreds = ddoutYpreds + [y_pred8_temp]
      totaMSE+=mse8
      indexSepDynamic +=2

    '''
    y_pred_temp = T.stack([y_pred1_temp, y_pred2_temp, y_pred3_temp, y_pred4_temp], axis=2) 
    y_pred_temp = y_pred_temp.flatten(3)# because of the stack, i guess, there's a 4th dimension created
    mse = T.mean((y_pred_temp - y.reshape((y.shape[0], y.shape[1],-1)))**2) # cause mse can be 26000
    '''
    #corr_in = corr_temp.reshape((x_shape[0]*x_shape[1], -1))
    #binary_in = binary_temp.reshape((x_shape[0]*x_shape[1], -1))

    recon = GMMdisagMulti(y_dim, y_in, theta_mu1_in, theta_sig1_in, coeff1_in, *argsGMM)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon = recon.reshape((x_shape[0], x_shape[1]))
    recon.name = 'gmm_out'

    '''
    recon5 = GMM(y_in[:,4, None], theta_mu5_in, theta_sig5_in, coeff5_in)
    recon5 = recon.reshape((x_shape[0], x_shape[1]))    
    '''
    recon_term = recon.sum(axis=0).mean()
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
    nll_upper_bound.name = 'nll_upper_bound'

    '''
    max_x = x.max()
    mean_x = x.mean()
    min_x = x.min()
    max_x.name = 'max_x'
    mean_x.name = 'mean_x'
    min_x.name = 'min_x'

    max_theta_mu = theta_mu_in.max()
    mean_theta_mu = theta_mu_in.mean()
    min_theta_mu = theta_mu_in.min()
    max_theta_mu.name = 'max_theta_mu'
    mean_theta_mu.name = 'mean_theta_mu'
    min_theta_mu.name = 'min_theta_mu'

    max_theta_sig = theta_sig_in.max()
    mean_theta_sig = theta_sig_in.mean()
    min_theta_sig = theta_sig_in.min()
    max_theta_sig.name = 'max_theta_sig'
    mean_theta_sig.name = 'mean_theta_sig'
    min_theta_sig.name = 'min_theta_sig'

    coeff_max = coeff_in.max()
    coeff_min = coeff_in.min()
    coeff_mean_max = coeff_in.mean(axis=0).max()
    coeff_mean_min = coeff_in.mean(axis=0).min()
    coeff_max.name = 'coeff_max'
    coeff_min.name = 'coeff_min'
    coeff_mean_max.name = 'coeff_mean_max'
    coeff_mean_min.name = 'coeff_mean_min'

    max_phi_sig = phi_sig_temp.max()
    mean_phi_sig = phi_sig_temp.mean()
    min_phi_sig = phi_sig_temp.min()
    max_phi_sig.name = 'max_phi_sig'
    mean_phi_sig.name = 'mean_phi_sig'
    min_phi_sig.name = 'min_phi_sig'

    max_prior_sig = prior_sig_temp.max()
    mean_prior_sig = prior_sig_temp.mean()
    min_prior_sig = prior_sig_temp.min()
    max_prior_sig.name = 'max_prior_sig'
    mean_prior_sig.name = 'mean_prior_sig'
    min_prior_sig.name = 'min_prior_sig'
    '''
    model.inputs = [x, mask, y, y_mask]
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
                   ddout=[nll_upper_bound, recon_term, kl_term, mse1, mae1]+ddoutMSEA+ddoutYpreds ,
                   indexSep=indexSepDynamic,
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

    mainloop = Training(
        name=pkl_name,
        data=Iterator(train_data, batch_size),
        model=model,
        optimizer=optimizer,
        cost=nll_upper_bound,
        outputs=[nll_upper_bound],
        extension=extension,
        lr_iterations=lr_iterations
    )
    mainloop.run()
    fLog = open(save_path+'/output.csv', 'w')
    fLog.write(str(lr_iterations)+"\n")
    fLog.write(str(appliances)+"\n")
    fLog.write(str(windows)+"\n")
    fLog.write("q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim\n")
    fLog.write("{},{},{},{},{},{}\n".format(q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim))
    fLog.write("epoch,log,kl,mse1,mse2,mse3,mse4,mse5,mse6,mse7,mse8,mae1,mae2,mae3,mae4,mae5,mae6,mae7,mae8\n")
    for i , item in enumerate(mainloop.trainlog.monitor['nll_upper_bound']):
      e,f,g,h,j,k,l,n,p,q,r,s,t,u =  '','','','','','','','','','','','','',''
      ep = mainloop.trainlog.monitor['epoch'][i]
      a = mainloop.trainlog.monitor['recon_term'][i]
      b = mainloop.trainlog.monitor['kl_term'][i]
      d = mainloop.trainlog.monitor['mse1'][i]
      m = mainloop.trainlog.monitor['mae1'][i]
      
      if (y_dim>1):
        e = mainloop.trainlog.monitor['mse2'][i]
        n = mainloop.trainlog.monitor['mae2'][i]
      if (y_dim>2):
        f = mainloop.trainlog.monitor['mse3'][i]
        p = mainloop.trainlog.monitor['mae3'][i]
      if (y_dim>3):
        g = mainloop.trainlog.monitor['mse4'][i]
        q = mainloop.trainlog.monitor['mae4'][i]
      if (y_dim>4):
        h = mainloop.trainlog.monitor['mse5'][i]
        r = mainloop.trainlog.monitor['mae5'][i]
      if (y_dim>5):
        j = mainloop.trainlog.monitor['mse6'][i]
        s = mainloop.trainlog.monitor['mae6'][i]
      if (y_dim>6):
        k = mainloop.trainlog.monitor['mse7'][i]
        t = mainloop.trainlog.monitor['mae7'][i]
      if (y_dim>7):
        l = mainloop.trainlog.monitor['mse8'][i]
        u = mainloop.trainlog.monitor['mae8'][i]
      fLog.write("{:d},{:2f},{:2f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
                  ep,a,b,d,e,f,g,h,j,k,l,m,n,p,q,r,s,t,u))


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