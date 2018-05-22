#import ipdb
import numpy as np
import theano
import theano.tensor as T
import datetime
import shutil
import os
import cPickle
import matplotlib.pyplot as plt

from cle.cle.cost import BiGMM, KLGaussianGaussian, GMM
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
from cle.cle.utils.op import Gaussian_sample, GMM_sample
from cle.cle.utils.gpu_op import concatenate

from VRNN_theano_version.datasets.dataport import Dataport
from VRNN_theano_version.datasets.dataport_utils import fetch_dataport

appliances = ['air1', 'furnace1', 'refrigerator1',  'clotheswasher1','drye1','dishwasher1', 'kitchenapp1', 'microwave1']
#appliancesNames = ['air1', 'furnace1', 'refrigerator1',  'clotheswasher1','drye1','dishwasher1', 'kitchenapp1', 'microwave1']
windows = {2859:("2015-01-01", "2016-01-01"),6990:("2015-01-01", "2016-01-01"),7951:("2015-01-01", "2016-01-01"),8292:("2015-01-01", "2016-01-01"),3413:("2015-01-01", "2016-01-01")}
#air: windows = {2859:("2015-06-25", "2015-11-01"),6990:("2015-06-01", "2015-11-01"),7951:("2015-06-01", "2015-11-01"),8292:("2015-06-01", "2015-11-01"),3413:("2015-06-01", "2015-11-01")}
#windows = {2859:("2015-01-01", "2016-01-01"),6990:("2015-01-01", "2016-01-01"),7951:("2015-01-01", "2016-01-01"),8292:("2015-01-01", "2016-01-01"),3413:("2015-01-01", "2016-01-01")}
listDates = {2859:['2015-08-26 07:57'],6990:['2015-10-15 08:18']}

def main(args):
    
    #theano.optimizer='fast_compile'
    #theano.config.exception_verbosity='high'

    trial = int(args['trial'])
    pkl_name = 'vrnn_gmm_%d' % trial
    channel_name = 'nll_upper_bound'

    data_path = args['data_path']
    save_path = args['save_path'] #+'/gmm/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    flgMSE = int(args['flgMSE'])

    period = int(args['period'])
    n_steps = int(args['n_steps'])
    stride_train = int(args['stride_train'])
    stride_test = n_steps# int(args['stride_test'])

    monitoring_freq = int(args['monitoring_freq'])
    epoch = int(args['epoch'])
    batch_size = int(args['batch_size'])
    x_dim = int(args['x_dim'])
    y_dim = int(args['y_dim'])
    flgAgg = int(args['flgAgg'])
    z_dim = int(args['z_dim'])
    rnn_dim = int(args['rnn_dim'])
    k = int(args['num_k']) #a mixture of K Gaussian functions
    lr = float(args['lr'])
    typeLoad = int(args['typeLoad'])
    debug = int(args['debug'])
    n_steps_val=n_steps

    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "saving pkl file '%s'" % pkl_name
    print "to the save path '%s'" % save_path
    print str(windows)

    q_z_dim = 180
    p_z_dim = 180
    p_x_dim = 200
    x2s_dim = 100
    z2s_dim = 150
    target_dim = k#x_dim #(x_dim-1)*k

    model = Model()
    Xtrain, ytrain, Xval, yval, Xtest, ytest, reader = fetch_dataport(data_path, windows, appliances,numApps=flgAgg, period=period,
                                              n_steps= n_steps, stride_train = stride_train, stride_test = stride_test,
                                              trainPer=0.6, valPer=0.2, testPer=0.2, typeLoad=typeLoad,
                                              flgAggSumScaled = 1, flgFilterZeros = 1)
    
    instancesPlot = {0:[5]} #for now use hard coded instancesPlot for kelly sampling

    train_data = Dataport(name='train',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         inputX=ytrain ,
                         labels=Xtrain)

    X_mean = train_data.X_mean
    X_std = train_data.X_std

    valid_data = Dataport(name='valid',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         X_mean=X_mean,
                         X_std=X_std,
                         inputX=yval,
                         labels = Xval)


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

    x_1 = FullyConnectedLayer(name='x_1',
                              parent=['x_t'],
                              parent_dim=[x_dim],
                              nout=x2s_dim,
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
               parent=['x_1', 'z_1'],
               parent_dim=[x2s_dim, z2s_dim],
               nout=rnn_dim,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

    '''
    dissag_pred = FullyConnectedLayer(name='disag_1',
                                  parent=['s_tm1'],
                                  parent_dim=[rnn_dim],
                                  nout=num_apps,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)
    '''
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

    theta_mu = FullyConnectedLayer(name='theta_mu',
                                   parent=['theta_1'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    theta_sig = FullyConnectedLayer(name='theta_sig',
                                    parent=['theta_1'],
                                    parent_dim=[p_x_dim],
                                    nout=target_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    coeff = FullyConnectedLayer(name='coeff',
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
             x_1, z_1, #dissag_pred,
             phi_1, phi_mu, phi_sig,
             prior_1, prior_mu, prior_sig,
             theta_1, theta_mu, theta_sig, coeff]#, corr, binary

    params = OrderedDict()

    for node in nodes:
        if node.initialize() is not None:
            params.update(node.initialize())

    params = init_tparams(params)

    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)

    def inner_val_fn(s_tm1):

        '''
        phi_1_t = phi_1.fprop([x_t, s_tm1], params)
        phi_mu_t = phi_mu.fprop([phi_1_t], params)
        phi_sig_t = phi_sig.fprop([phi_1_t], params)
        '''
        prior_1_t = prior_1.fprop([s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(prior_mu_t, prior_sig_t)
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)
        theta_mu_t = theta_mu.fprop([theta_1_t], params)
        theta_sig_t = theta_sig.fprop([theta_1_t], params)
        coeff_t = coeff.fprop([theta_1_t], params)
        
        pred_t = GMM_sample(theta_mu_t, theta_sig_t, coeff_t) #Gaussian_sample(theta_mu_t, theta_sig_t)
        pred_1_t = x_1.fprop([pred_t], params)

        s_t = rnn.fprop([[pred_1_t, z_1_t], [s_tm1]], params)

        return s_t, pred_t,  z_t, theta_1_t, theta_mu_t, theta_sig_t, coeff_t
        # prior_mu_temp_val, prior_sig_temp_val
    ((s_temp_val, prediction_val, z_t_temp_val, theta_1_temp_val, theta_mu_temp_val, theta_sig_temp_val, coeff_temp_val), updates_val) =\
        theano.scan(fn=inner_val_fn , n_steps=n_steps_val, #already 1 subtracted if doing next step
                    outputs_info=[s_0, None, None,  None, None, None, None])

    for k, v in updates_val.iteritems():
        k.default_update = v

    def inner_fn(x_t, s_tm1):

        phi_1_t = phi_1.fprop([x_t, s_tm1], params)
        phi_mu_t = phi_mu.fprop([phi_1_t], params)
        phi_sig_t = phi_sig.fprop([phi_1_t], params)

        prior_1_t = prior_1.fprop([s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(phi_mu_t, phi_sig_t)
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)
        theta_mu_t = theta_mu.fprop([theta_1_t], params)
        theta_sig_t = theta_sig.fprop([theta_1_t], params)

        coeff_t = coeff.fprop([theta_1_t], params)
        #corr_t = corr.fprop([theta_1_t], params)
        #binary_t = binary.fprop([theta_1_t], params)

        pred = GMM_sample(theta_mu_t, theta_sig_t, coeff_t) #Gaussian_sample(theta_mu_t, theta_sig_t)

        s_t = rnn.fprop([[x_t, z_1_t], [s_tm1]], params)

        #y_pred = dissag_pred.fprop([s_t], params)

        return s_t, phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, z_t,  z_1_t, theta_1_t, theta_mu_t, theta_sig_t, coeff_t, pred#, y_pred
        #corr_temp, binary_temp
    ((s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp,z_t_temp, z_1_temp, theta_1_temp, theta_mu_temp, theta_sig_temp, coeff_temp, prediction), updates) =\
        theano.scan(fn=inner_fn,
                    sequences=[x_1_temp ],
                    outputs_info=[s_0, None, None, None, None, None, None,  None, None, None, None, None])

    
    for k, v in updates.iteritems():
        k.default_update = v
    
    s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)# seems like this is for creating an additional dimension to s_0
    '''
    theta_1_temp = theta_1.fprop([z_1_temp, s_temp], params)
    theta_mu_temp = theta_mu.fprop([theta_1_temp], params)
    theta_sig_temp = theta_sig.fprop([theta_1_temp], params)
    coeff_temp = coeff.fprop([theta_1_temp], params)
    corr_temp = corr.fprop([theta_1_temp], params)
    binary_temp = binary.fprop([theta_1_temp], params)
    '''

    s_temp.name = 'h'#gisse
    z_1_temp.name = 'z2'#gisse
    z_t_temp.name = 'z'
    theta_mu_temp.name = 'mu'
    theta_sig_temp.name = 'sig'
    coeff_temp.name = 'coeff'
    
    prediction.name = 'Prediction-'+str(appliances[flgAgg][:-1])
    mse = T.mean((prediction - x)**2) # As axis = None is calculated for all
    mae = T.mean( T.abs_(prediction - x) )
    mse.name = 'mse'
    mae.name = 'mae'
    x_in = x.reshape((batch_size*n_steps,-1))

    kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)

    x_shape = x.shape
    
    theta_mu_in = theta_mu_temp.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig_in = theta_sig_temp.reshape((x_shape[0]*x_shape[1], -1))
    coeff_in = coeff_temp.reshape((x_shape[0]*x_shape[1], -1))
    #corr_in = corr_temp.reshape((x_shape[0]*x_shape[1], -1))
    #binary_in = binary_temp.reshape((x_shape[0]*x_shape[1], -1))

    recon = GMM(x_in, theta_mu_in, theta_sig_in, coeff_in)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon = recon.reshape((x_shape[0], x_shape[1]))
    recon.name = 'gmm_out'
    
    #recon = recon * mask
    
    recon_term = recon.sum(axis=0).mean()
    recon_term.name = 'recon_term'

    #kl_temp = kl_temp * mask
    
    kl_term = kl_temp.sum(axis=0).mean()
    kl_term.name = 'kl_term'

    nll_upper_bound = recon_term + kl_term #+ mse
    if (flgMSE):
      nll_upper_bound = nll_upper_bound + mse
    nll_upper_bound.name = 'nll_upper_bound'

    ############## TEST  ###############
    theta_mu_in_val = theta_mu_temp_val.reshape((batch_size*n_steps, -1))
    theta_sig_in_val = theta_sig_temp_val.reshape((batch_size*n_steps, -1))
    coeff_in_val = coeff_temp_val.reshape((batch_size*n_steps,-1))


    pred_in = prediction_val.reshape((batch_size*n_steps,-1))
    recon_val = GMM(pred_in, theta_mu_in_val, theta_sig_in_val, coeff_in_val)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon_val = recon_val.reshape((batch_size, n_steps))
    recon_val.name = 'gmm_out_val'

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
                   ddout=[nll_upper_bound, recon_term, kl_term, mse, mae,
                          prediction],
                   indexSep=5,
                   indexDDoutPlot = [(0,theta_mu_temp), (2, z_t_temp), (3,prediction)],
                   instancesPlot = instancesPlot, #{0:[4,20],2:[5,10]},#, 80,150
                   data=[Iterator(valid_data, batch_size)],
                   savedFolder = save_path),
        Picklize(freq=monitoring_freq, path=save_path),
        EarlyStopping(freq=monitoring_freq, path=save_path, channel=channel_name),
        WeightNorm()
    ]

    lr_iterations = {0:lr, 30:(lr/10)}#, 150:(lr/10), 270:(lr/100), 370:(lr/1000)

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
    mainloop.run()

    test_fn = theano.function(inputs=[],
                          outputs=[prediction_val, recon_val],
                          updates=updates_val#, allow_input_downcast=True, on_unused_input='ignore'
                          )

    outputGeneration = test_fn()
    #{0:[4,20], 2:[5,10]} 
    '''
    plt.figure(1)
    plt.plot(np.transpose(outputGeneration[0],[1,0,2])[5])
    plt.savefig(save_path+"/vrnn_dis_generated_z_0-4.ps")

    plt.figure(2)
    plt.plot(np.transpose(outputGeneration[1],[1,0,2])[5])
    plt.savefig(save_path+"/vrnn_dis_generated_s_0-4.ps")

    plt.figure(3)
    plt.plot(np.transpose(outputGeneration[2],[1,0,2])[5])
    plt.savefig(save_path+"/vrnn_dis_generated_theta_0-4.ps")
    '''
    plt.figure(1)
    plt.plot(np.transpose(outputGeneration[0],[1,0,2])[2])
    plt.savefig(save_path+"/vrnn_dis_generated_pred_0-2.ps")

    plt.figure(2)
    plt.plot(np.transpose(outputGeneration[0],[1,0,2])[10])
    plt.savefig(save_path+"/vrnn_dis_generated_pred_0-10.ps")

    plt.figure(3)
    plt.plot(np.transpose(outputGeneration[0],[1,0,2])[15])
    plt.savefig(save_path+"/vrnn_dis_generated_pred_0-15.ps")

    testLogLike = np.asarray(outputGeneration[1]).mean()

    fLog = open(save_path+'/output.csv', 'w')
    fLog.write(str(lr_iterations)+"\n")
    fLog.write(str(windows)+"\n")
    fLog.write("Test-log-likelihood\n")
    fLog.write("{}\n".format(testLogLike))
    fLog.write("q_z_dim,p_z_dim,p_x_dim,x2s_dim,z2s_dim\n")
    fLog.write("{},{},{},{},{}\n".format(q_z_dim,p_z_dim,p_x_dim,x2s_dim,z2s_dim))
    fLog.write("epoch,log,kl,nll_upper_bound,mse,mae\n")
    for i , item in enumerate(mainloop.trainlog.monitor['nll_upper_bound']):
      f = mainloop.trainlog.monitor['epoch'][i]
      a = mainloop.trainlog.monitor['recon_term'][i]
      b = mainloop.trainlog.monitor['kl_term'][i]
      c = mainloop.trainlog.monitor['nll_upper_bound'][i]
      d = mainloop.trainlog.monitor['mse'][i]
      e = mainloop.trainlog.monitor['mae'][i]
      fLog.write("{:d},{:.2f},{:.2f},{:.2f},{:.3f},{:.3f}\n".format(f,a,b,c,d,e))
    fLog.close()
    
    f = open(save_path+'/outputRealGeneration.pkl', 'wb')
    cPickle.dump(outputGeneration, f, -1)
    f.close()


if __name__ == "__main__":

    import sys, time
    if len(sys.argv) > 1:
        config_file_name = sys.argv[-1]
    else:
        config_file_name = 'config.txt'

    f = open(config_file_name, 'r')
    lines = f.readlines()
    params = OrderedDict()

    for line in lines:
        line = line.split('\n')[0]
        param_list = line.split(' ')
        param_name = param_list[0]
        param_value = param_list[1]
        params[param_name] = param_value

    params['save_path'] = params['save_path']+'/gmm/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")+'_app'+params['flgAgg']
    os.makedirs(params['save_path'])
    shutil.copy('config.txt', params['save_path']+'/config.txt')

    main(params)
