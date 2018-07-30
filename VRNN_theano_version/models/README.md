############# SET PARAMTERS AND VARIABLES  #########################################
Parameters are set in several places:
- config-AE.txt: some of the paramters that not that obvious are
	- monitoring_freq: number of training batch after testing in validation set. It depends on the size of the training set
	- flgAgg: -1 sample original x, 0: first appliance - 1 second appliance ... 4: 5th appliance
	- genCase: generates from reconstruction for same time step (0) or generation for next step (1)
	- typeLoad: 
		0 for original load where the total set is divided in training, validation and testing in an specific order.
		1 Building instances randomly, without any overlap and assigning specific number to train, test and val
	- stride_train:
		1: when typeLoad is 0
		n_steps: when the typeLoad is 1
- Hardcode: buildings, APPLIANCES (we call it by specifying the number of appliances we want at loadCSVdataVRNN
), windows of time and instances to plot

- vrnn-disall-priorXt.py adds the prior be based on xt (aggregated of the current state) and also corrects (Because the encoder wasn't receiveing the yt)

- test_vrnn_dis1 accepts both pkl file that store scheduling and non scheduling models.