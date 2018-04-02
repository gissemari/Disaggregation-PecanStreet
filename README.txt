###############  COMPONENTS ############

- dataSet.py is the loader of the Dataport dataset which is obtained from Pecan-Street dataport and user registration. Follos Sabina Thompson's instructions.
- autoEncoder_tf.py
- lstm_tf.py
- VRNN_theano/models/* contains all the vrnn models based on the paper: Recurrent Latent Variable Model for Sequential Data (Junyoung Chung, Kyle Kastner ...)

###############   SOME PRE KNOWLEDGE ABOUT THE MODEL   ###############
- It uses the models/cle (can be referenced from the UKDALE/.../cle)
- It runs with python2.7
- It uses the latenVar 

##############        RUNNING AND PARAMTERS   ####################

Most of the parameters are similar through all vrnn models:
- data_path is the path of data files: .../PecanStreet-dataport/datasets
- save_path is the folder where the output is going to be saved, usually .../VRNN_theano_version/output
- flgMSE 0 for not include mse in the cost and 1 otherwise
- monitoring_freq 310
epoch 100
batch_size 250
x_dim 1
y_dim 8
z_dim 80
num_k 20
rnn_dim 100
lr 0.001
debug 0
period 6
n_steps 80
stride_train 80
stride_test 80