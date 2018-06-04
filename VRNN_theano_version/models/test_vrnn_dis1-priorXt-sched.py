#import ipdb
import numpy as np
import theano
import theano.tensor as T
import datetime
import shutil
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
import cPickle

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

building = 2859
appliances = ['air1', 'furnace1', 'refrigerator1',  'clotheswasher1','drye1','dishwasher1', 'kitchenapp1', 'microwave1']
#windows = {building:("2015-01-01", "2015-01-01")}#, 2859:("2015-01-01", "2016-01-01"), 7951:("2015-01-01", "2016-01-01"),8292:("2015-01-01",  "2016-01-01"),3413:("2015-01-01", "2016-01-01")}#3413:("2015-01-01", "2015-12-31")
# dishwasher: windows = {6990:("2015-02-12", "2016-01-01"), 2859:("2015-02-01", "2015-12-15"), 7951:("2015-01-01", "2016-01-01"),8292:("2015-01-04",  "2016-01-01"),3413:("2015-01-01", "2015-12-25")}#3413:("2015-01-01", "2015-12-31")
# drye: windows = {2859:("2015-01-01", "2016-01-01"),6990:("2015-01-01", "2016-01-01"),7951:("2015-01-01", "2016-01-01"),8292:("2015-01-01",  "2016-01-01"),3413:("2015-01-20", "2015-12-31")}#3413:("2015-01-01", "2015-12-31")
windows = {2859:("2015-01-01", "2016-01-01"),6990:("2015-01-01", "2016-01-01"),7951:("2015-01-01", "2016-01-01"),8292:("2015-01-01",  "2016-01-01"),3413:("2015-01-01", "2016-01-01")}
listDates = {2859:['2015-08-26 07:57'],6990:['2015-10-15 08:18']}

def main(args):
    
    #theano.optimizer='fast_compile'
    #theano.config.exception_verbosity='high'

    trial = int(args['trial'])
    pkl_name = 'dp_dis1-sch_%d' % trial
    channel_name = 'mae'

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
    kSchedSamp = int(args['kSchedSamp'])

    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "saving pkl file '%s'" % pkl_name
    print "to the save path '%s'" % save_path

    q_z_dim = 150
    p_z_dim = 150
    p_x_dim = 150#250
    x2s_dim = 100#250
    y2s_dim = 100
    z2s_dim = 100#150
    target_dim = k#x_dim #(x_dim-1)*k

    model = Model()
    Xtrain, ytrain, Xval, yval, Xtest, ytest, reader = fetch_dataport(data_path, windows, appliances,numApps=flgAgg, period=period,
                                              n_steps= n_steps, stride_train = stride_train, stride_test = stride_test,
                                              trainPer=0.6, valPer=0.2, testPer=0.2, typeLoad=typeLoad,
                                              flgAggSumScaled = 1, flgFilterZeros = 1)
    print(reader.stdTrain, reader.meanTrain)
    instancesPlot = {0:[4], 2:[10]} #for now use hard coded instancesPlot for kelly sampling

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
    scheduleSamplingMask  = T.fvector('schedMask')
 
    x.name = 'x_original'
    if debug:
        x.tag.test_value = np.zeros((15, batch_size, x_dim), dtype=np.float32)
        temp = np.ones((15, batch_size), dtype=np.float32)
        temp[:, -2:] = 0.
        mask.tag.test_value = temp


    fmodel = open('dp_dis1-sch_1_best.pkl', 'rb')
    mainloop = cPickle.load(fmodel)
    fmodel.close()

    #attrs = vars(mainloop)
    #print ', '.join("%s: %s" % item for item in attrs.items())
    """names = [x.name for x in mainloop.model.nodes]
    print(names)
    print(mainloop.model.nodes)"""

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
    theta_mu = mainloop.model.nodes[11]
    theta_sig = mainloop.model.nodes[12]
    coeff = mainloop.model.nodes[13]

    nodes = [rnn,
         x_1, y_1, z_1, #dissag_pred,
         phi_1, phi_mu, phi_sig,
         prior_1, prior_mu, prior_sig,
         theta_1, theta_mu, theta_sig, coeff]#, corr, binary

    params = mainloop.model.params
    """params = OrderedDict()

    for node in nodes:
        if node.initialize() is not None:
            params.update(node.initialize())"""

    #params = init_tparams(params)

    """OrderedDict([('W_x_1__rnn', W_x_1__rnn), ('W_z_1__rnn', W_z_1__rnn), ('W_y_1__rnn', W_y_1__rnn), 
    ('U_rnn__rnn', U_rnn__rnn), ('b_rnn', b_rnn), ('W_x_t__x_1', W_x_t__x_1), ('b_x_1', b_x_1), 
    ('W_y_t__y_1', W_y_t__y_1), ('b_y_1', b_y_1), ('W_z_t__z_1', W_z_t__z_1), ('b_z_1', b_z_1), 
    ('W_x_1__phi_1', W_x_1__phi_1), ('W_s_tm1__phi_1', W_s_tm1__phi_1), ('W_y_1__phi_1', W_y_1__phi_1), 
    ('b_phi_1', b_phi_1), ('W_phi_1__phi_mu', W_phi_1__phi_mu), ('b_phi_mu', b_phi_mu), 
    ('W_phi_1__phi_sig', W_phi_1__phi_sig), ('b_phi_sig', b_phi_sig), ('W_x_1__prior_1', W_x_1__prior_1), 
    ('W_s_tm1__prior_1', W_s_tm1__prior_1), ('b_prior_1', b_prior_1), ('W_prior_1__prior_mu', W_prior_1__prior_mu), 
    ('b_prior_mu', b_prior_mu), ('W_prior_1__prior_sig', W_prior_1__prior_sig), ('b_prior_sig', b_prior_sig), 
    ('W_z_1__theta_1', W_z_1__theta_1), ('W_s_tm1__theta_1', W_s_tm1__theta_1), ('b_theta_1', b_theta_1), 
    ('W_theta_1__theta_mu', W_theta_1__theta_mu), ('b_theta_mu', b_theta_mu), ('W_theta_1__theta_sig', W_theta_1__theta_sig), 
    ('b_theta_sig', b_theta_sig), ('W_theta_1__coeff', W_theta_1__coeff), ('b_coeff', b_coeff)])"""

    """w1 = params['W_x_1__rnn']
    w2 = params['W_phi_1__phi_sig']
    w3 = params['W_theta_1__coeff']

    print("Initialized W matricies:")
    print("W1:",w1.get_value())
    print("W2:",w2.get_value())
    print("W3:",w3.get_value())
    print("\n\n--------------------------------\n\n")"""

    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)
    y_1_temp = y_1.fprop([y], params)


    def inner_fn_val(x_t, s_tm1):

        prior_1_t = prior_1.fprop([x_t,s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(prior_mu_t, prior_sig_t)
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)
        theta_mu_t = theta_mu.fprop([theta_1_t], params)
        theta_sig_t = theta_sig.fprop([theta_1_t], params)

        coeff_t = coeff.fprop([theta_1_t], params)

        pred_t = GMM_sample(theta_mu_t, theta_sig_t, coeff_t) #Gaussian_sample(theta_mu_t, theta_sig_t)
        pred_1_t = y_1.fprop([pred_t], params)
        s_t = rnn.fprop([[x_t, z_1_t, pred_1_t], [s_tm1]], params)
        #y_pred = dissag_pred.fprop([s_t], params)

        return s_t, prior_mu_t, prior_sig_t, theta_mu_t, theta_sig_t, coeff_t, pred_t#, y_pred
        #corr_temp, binary_temp
    ((s_temp_val, prior_mu_temp_val, prior_sig_temp_val, theta_mu_temp_val, theta_sig_temp_val, coeff_temp_val, prediction_val), updates_val) =\
        theano.scan(fn=inner_fn_val,
                    sequences=[x_1_temp],
                    outputs_info=[s_0, None, None, None,  None, None, None])

    for k, v in updates_val.iteritems():
        k.default_update = v

    s_temp_val = concatenate([s_0[None, :, :], s_temp_val[:-1]], axis=0)

   
    x_shape = x.shape

    ######################## TEST (GENERATION) TIME
    prediction_val.name = 'generated__'+str(flgAgg)
    mse_val = T.mean((prediction_val - y)**2) # As axis = None is calculated for all
    mae_val = T.mean( T.abs_(prediction_val - y) )

    mse_val.name = 'mse_val'
    mae_val.name = 'mae_val'
    pred_in_val = y.reshape((y.shape[0]*y.shape[1],-1))

    theta_mu_in_val = theta_mu_temp_val.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig_in_val = theta_sig_temp_val.reshape((x_shape[0]*x_shape[1], -1))
    coeff_in_val = coeff_temp_val.reshape((x_shape[0]*x_shape[1], -1))

    recon_val = GMM(pred_in_val, theta_mu_in_val, theta_sig_in_val, coeff_in_val)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon_val = recon_val.reshape((x_shape[0], x_shape[1]))
    recon_val.name = 'gmm_out_val'

    recon_term_val= recon_val.sum(axis=0).mean()
    recon_term_val.name = 'recon_term_val'

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
                   #ddout=[nll_upper_bound, recon_term, kl_term, mse, mae, prediction],
                   indexSep=5,
                   instancesPlot = instancesPlot, #{0:[4,20],2:[5,10]},#, 80,150
                   data=[Iterator(valid_data, batch_size)],
                   savedFolder = save_path),
        Picklize(freq=monitoring_freq, path=save_path),
        EarlyStopping(freq=monitoring_freq, path=save_path, channel=channel_name),
        WeightNorm()
    ]


    """params = OrderedDict()

    for node in mainloop.model.nodes:
        if node.initialize() is not None:
            params.update(node.initialize())

    params = init_tparams(params)"""

    """ w11 = params['W_x_1__rnn']
    w21 = params['W_phi_1__phi_sig']
    w31 = params['W_theta_1__coeff']

    print("Pickle W matricies:")
    print("W1:",w11.get_value())
    print("W2:",w21.get_value())
    print("W3:",w31.get_value())"""

    """mainloop.restore(
      name=pkl_name,
      data=Iterator(train_data, batch_size),
      model=model,
      optimizer=optimizer,
      #cost=nll_upper_bound,
      #outputs=[recon_term, kl_term, nll_upper_bound, mse, mae],
      n_steps = n_steps,
      extension=extension,
      #lr_iterations=lr_iterations,
      k_speedOfconvergence=kSchedSamp
    )"""


    data=Iterator(test_data, batch_size)

    test_fn = theano.function(inputs=[x, y],#[x, y],
                              #givens={x:Xtest},
                              #on_unused_input='ignore',
                              #z=( ,200,1)
                              allow_input_downcast=True,
                              outputs=[prediction_val, recon_term_val, mse_val, mae_val]#prediction_val, mse_val, mae_val
                              ,updates=updates_val#, allow_input_downcast=True, on_unused_input='ignore'
                              )
    testOutput = []
    numBatchTest = 0
    for batch in data:
      outputGeneration = test_fn(batch[0], batch[2])#(20, 220, 1)
      testOutput.append(outputGeneration[1:])

      plt.figure(4)
      plt.plot(np.transpose(outputGeneration[0],[1,0,2])[4])
      plt.plot(np.transpose(batch[2],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_RealAndPred_0-4".format(numBatchTest))
      plt.clf()

      plt.figure(4)
      plt.plot(np.transpose(batch[0],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_Realagg_0-4".format(numBatchTest))
      plt.clf()
      numBatchTest+=1

    testOutput = np.asarray(testOutput)
    print(testOutput.shape)
    recon_test  = testOutput[:, 0].mean()
    mse_test = testOutput[:, 1].mean()
    mae_test = testOutput[:, 2].mean()
    #mseUnNorm_test = testOutput[:, 3].mean()
    #maeUnNorm_test = testOutput[:, 4].mean()

    fLog = open(save_path+'/output.csv', 'w')
    #fLog.write(str(lr_iterations)+"\n")
    fLog.write(str(windows)+"\n")
    fLog.write("logTest,mseTest,maeTest, mseTestUnNorm, maeTestUnNorm\n")
    fLog.write("{},{},{}\n".format(recon_test,mse_test,mae_test))
    fLog.write("q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim\n")
    fLog.write("{},{},{},{},{},{}\n".format(q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim))
    header = "epoch,log,kl,mse,mae\n"
    fLog.write(header)
    for i , item in enumerate(mainloop.trainlog.monitor['recon_term']):
      f = mainloop.trainlog.monitor['epoch'][i]
      a = mainloop.trainlog.monitor['recon_term'][i]
      b = mainloop.trainlog.monitor['kl_term'][i]
      d = mainloop.trainlog.monitor['mse'][i]
      e = mainloop.trainlog.monitor['mae'][i]
      fLog.write("{:d},{:.2f},{:.2f},{:.3f},{:.3f}\n".format(f,a,b,d,e))

if __name__ == "__main__":

    import sys, time
    if len(sys.argv) > 1:
        config_file_name = sys.argv[-1]
    else:
        config_file_name = 'config_AE.txt'

    f = open(config_file_name, 'r')
    lines = f.readlines()
    params = OrderedDict()

    for line in lines:
        line = line.split('\n')[0]
        param_list = line.split(' ')
        param_name = param_list[0]
        param_value = param_list[1]
        params[param_name] = param_value

    params['save_path'] = params['save_path']+'/gmmAE/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")+'_app'+params['flgAgg']
    os.makedirs(params['save_path'])
    shutil.copy('config_AE.txt', params['save_path']+'/config_AE.txt')

    main(params)
