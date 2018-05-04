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
- monitoring_freq: number of training batches after which perform a validation process
- epoch: number of iterations
- batch_size
- x_dim: aggregated signal (1)
- y_dim: disaggregated signal (number of appliances)
- z_dim: units in z layer
- num_k: number of mixture gaussian distributions
- rnn_dim: number of units for the recurrent layer
- lr: learning rate
- period: not used anymore because the sampling is already every one minute and we don't want lower frequency
- n_steps: sequence length
- stride_train: stride or space between instances in the training set
- stride_test: stride or space between instances in the testing set
- typeLoad: 
	- 0 for original load where the total set is divided in training, validation and testing in an specific order.
	- 2 Building instances randomly, without any overlap and assigning specific number to train, test and val
	- 1 (not used for this dataset. For ukdale: kelly's load)
- stride_train:
	- 1: when typeLoad is 0 (because it will use 1 to have overlapped instances in the training, but will use n_steps for the stride in the testing instances)
	- n_steps: when the typeLoad is 1 (because as we randomly choose instances, we need to guarantee that the testing instances or any part of them are also part of the training set)

- Hardcode: buildings, APPLIANCES (we call it by specifying the number of appliances we want at loadCSVdataVRNN
), windows of time and instances to plot

