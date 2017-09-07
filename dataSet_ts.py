import numpy as np
import pandas as pd

class ReaderTS(object):

	def __init__(self, numDim,  strd_in, strd_out):
		self.numDim = numDim
		self.stride_input = strd_in
		self.stride_output = strd_out

	@staticmethod
	def rnn_data( data, idxStart, idxEnd, time_steps, stride, labels=False):
	    """
	    creates new data frame based on previous observation
	      * example:
	        l = [1, 2, 3, 4, 5]
	        time_steps = 2
	        -> labels == False [[1, 2], [2, 3], [3, 4]]
	        -> labels == True [3, 4, 5]
	    """
	    rnn_df = []
	    target_apps = ['air1', 'furnace1', 'refrigerator1',  'clotheswasher1','drye1','dishwasher1', 'kitchenapp1', 'microwave1']
	    for i in range(idxStart, idxEnd, stride): # not a-b because we take all the examples
	        if labels:
	            try:
	            	aux = data.iloc[i:i + time_steps,1:].as_matrix()
	            	if (aux.shape[0] == time_steps):
	                	rnn_df.append(np.transpose(aux))
	            except AttributeError:
	            	aux = data.iloc[i:i + time_steps,1:]
	            	if (aux.shape[0] == time_steps):
	                	rnn_df.append(np.transpose(aux))
	        else:
	            data_ = data.iloc[i:i + time_steps,0].as_matrix()
	            aux_len = len(data_.shape)
	            if aux_len > 1:
	            	aux = data_
	            	#print aux_len  
	            else:
	            	aux = [[i] for i in data_]
	            	#print aux
	            if (data_.shape[0]==time_steps): ### Because the last instance not necessarily has the number of time steps but the remains
	            	rnn_df.append(np.transpose(aux)) ##### rows: 1 appliances - columns: time
	    #print ("labels {} shape {}".format(labels, len(rnn_df)))
	    return np.array(rnn_df, dtype=np.float32)

	@staticmethod
	def split_data(data, time_steps, val_size=0.1, test_size=0.1):
	    """
	    splits data to training, validation and testing parts
	    """
	    completeInstances = len(data.index)#data[0].count()#data.shape[0] - time_steps # 500000#
	    print (completeInstances)
	    ntest = int(round( completeInstances* (1 - test_size)))
	    nval = int(round(data[:ntest].shape[0] * (1 - val_size)))

	    #Sabina 50% - 25% - 25%
	    #print completeInstances, ntest, nval
	    return nval, ntest, completeInstances

	@staticmethod
	def prepare_data(filePath, home,  time_steps, stride_input, stride_output, labels=False, val_size=0.1, test_size=0.1):
	    """
	    Given the number of `time_steps` and some data,
	    prepares training, validation and test data for an lstm cell.
	    """
	    #filePath should be /mnt/data/Disaggregation-Pecan/
	    imagepath=filePath+"dataid_{}_minute.h5".format(home) #imagepath="dataport-2016-minute-8app.csv"
	    imagearray  = pd.HDFStore(imagepath)
	    #print imagearray.keys()
	    data=imagearray['df']

	    split1, split2, split3 = ReaderTS.split_data(data, time_steps, val_size, test_size)
	    return ReaderTS.rnn_data(data, 0, split1, time_steps, stride_input, labels=False),\
	           ReaderTS.rnn_data(data, split1, split2, time_steps, stride_output, labels=False),\
				ReaderTS.rnn_data(data, split2, split3, time_steps, stride_output, labels=False),\
				ReaderTS.rnn_data(data, 0, split1, time_steps, stride_input, labels=True),\
	            ReaderTS.rnn_data(data, split1, split2, time_steps, stride_output, labels=True),\
				ReaderTS.rnn_data(data, split2, split3, time_steps, stride_output, labels=True)

	def load_csvdata(self,filePath,time_steps,  seperate=False):
		#stamps 1451606400 - 1483228800: all 2016
		homes =[2859,8292,7951,3413]#, wrong6990
		#train_x,val_x,test_x, train_y, val_y, test_y = ReaderTS.prepare_data(homes, time_steps, self.stride_input, self.stride_output, val_size=0.25, test_size = 0.25)
				
		all_homes =[[ReaderTS.prepare_data(filePath, home, time_steps, self.stride_input, self.stride_output, val_size=0.25, test_size = 0.25)] for home in homes]

		a = np.squeeze(all_homes,axis=0)
		train_x = np.concatenate(a[:,0])#a[:,0]
		val_x = np.concatenate(a[:,1])
		test_x = np.concatenate(a[:,2])
		train_y = np.concatenate(a[:,3])
		val_y = np.concatenate(a[:,4])
		test_y = np.concatenate(a[:,5])
		print train_x.shape, train_y.shape, test_x.shape, test_y.shape

		return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)		
