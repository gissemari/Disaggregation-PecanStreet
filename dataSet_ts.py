import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import datetime
from sklearn import preprocessing

offValues = {2859:{'air1':0.2, 'furnace1':0.2, 'refrigerator1':0.1, 'clotheswasher1':0.2, 
                    'drye1':1, 'dishwasher1':0, 'kitchenapp1':0.005,  'microwave1':0.25},
             6990:{'air1':1, 'furnace1':0.3, 'refrigerator1':0.1, 'clotheswasher1':0.2, 
                    'drye1':2, 'dishwasher1':0.5, 'kitchenapp1':0.1,  'microwave1':0.05},
             7951:{'air1':0.5, 'furnace1':0.01, 'refrigerator1':0.1, 'clotheswasher1':0.1, 
                    'drye1':1, 'dishwasher1':0.4, 'kitchenapp1':0.2,  'microwave1':0.5},
             8292:{'air1':0.3, 'furnace1':0.2, 'refrigerator1':0.2, 'clotheswasher1':0.05, 
                    'drye1':1, 'dishwasher1':0.5, 'kitchenapp1':0.01,  'microwave1':0.5},
             3413:{'air1':0.3, 'furnace1':0.3, 'refrigerator1':0.1, 'clotheswasher1':0.1, 
                    'drye1':1, 'dishwasher1':0.5, 'kitchenapp1':0.4,  'microwave1':0.25}}

#'air1', 'furnace1','refrigerator1', 'clotheswasher1','drye1','dishwasher1', 'kitchenapp1','microwave1'

'''
offValues = {2859:{0:0.2,1:0.2,2:0.1,3:0.2,4:1,5:0,6:0.005,7:0.25},
             6990:{0:1,1:0.3,2:0.1,3:0.2,4:2,5:0.5,6:0.1,7:0.05},
             7951:{0:0.5,1:0.01,2:0.1,3:0.01,4:1,5:0.4,6:0.2,7:0.5},
             8292:{0:0.3,1:0.2,2:0.2,3:0.05,4:1,5:0.5,6:0.01,7:0.5},
             3413:{0:0.3,1:0.3,2:0.1,3:0.1,4:1,5:0.5,6:0.4,7:0.25}}
'''
class ReaderTS(object):

    def __init__(self, windows, appliances, time_steps, strd_in, strd_out, 
                    sample_period, flgAggSumScaled=0, flgFilterZeros = 0, 
                    flgScaling=0, trainPer=0.5, valPer=0.25, testPer=0.25):
        self.stride_input = strd_in
        self.stride_output = strd_out
        self.time_steps = time_steps
        self.pTrain = trainPer
        self.pVal = valPer
        self.pTest = testPer
        self.listAppliances = appliances
        self.windows = windows
        self.sample_period = sample_period
        self.flgScaling = flgScaling
        self.flgAggSumScaled = flgAggSumScaled
        self.flgFilterZeros = flgFilterZeros
        self.bin_edges = None
        self.idxFiltered = []
        self.meanTrain = 0
        self.stdTrain = 0
        assert (trainPer+valPer+testPer) == 1
        
    def rnn_data( self, data, idxStart, idxEnd, apps, stride, labels=False):
        rnn_df = []

        if labels:
            data = data[apps].values
            arrShape = [0,self.time_steps,len(apps)]
        else:
            data = data['use'].values
            arrShape = [0,self.time_steps]

        for i in range(idxStart, idxEnd, stride): # not a-b because we take all the examples
            seqY = data[i:i + self.time_steps]
            if (seqY.shape[0] == self.time_steps):
                rnn_df.append(seqY)
            else:
                continue #pad

        if (len(rnn_df)!=0):
            result = np.array(rnn_df) #shape (batch, x:1/y:apps, lengthSeq)
            return np.squeeze(result) # make it two dimensions if we are just selecting one appliance
        else:
            return np.empty(arrShape)
    
    def calculate_split(self, data):
        '''
        Spliting the data into training, validation and test sets according to the percentage values
        '''
        pRest = self.pVal + self.pTest
        completeInstances = len(data.index)
        print(completeInstances)
        indexVal = int(round((self.pTrain*self.stride_input*completeInstances / ( pRest*self.stride_output +self.pTrain*self.stride_input))))
        indexTest = int(indexVal + self.pVal*(completeInstances - indexVal)/pRest)

        #Sabina 50% - 25% - 25%
        return indexVal,indexTest,completeInstances
    
    def split_data(self, dataBuild, indexVal, indexTest, indexEnd,apps):
        train_y = self.rnn_data(dataBuild, 0, (indexVal), apps, self.stride_input, labels=True)
        val_y   = self.rnn_data(dataBuild, indexVal, indexTest, apps, self.stride_output, labels=True)
        test_y  = self.rnn_data(dataBuild, indexTest, indexEnd, apps, self.stride_output, labels=True)
        return train_y, val_y, test_y

    def convert3D(self, dataBuild, indexEnd):
        return self.rnn_data(dataBuild, 0, indexEnd, self.listAppliances, self.stride_input, labels=True)

    def scaling(self, train, val, test, newShape, thirdDim):
            ##### scaling
        flat_train = np.reshape(train,[-1,thirdDim])
        flat_val = np.reshape(val,[-1,thirdDim])
        flat_test = np.reshape(test,[-1,thirdDim])

        scalerY = preprocessing.StandardScaler().fit(flat_train)
        #mean-> mean and scale->std
        self.meanTrain = scalerY.mean_
        self.stdTrain = scalerY.scale_

        train = scalerY.transform(flat_train) if flat_train.shape[0]!=0 else train
        train = np.reshape(train, newShape)
        val = scalerY.transform(flat_val) if flat_val.shape[0]!=0 else val
        val = np.reshape(val, newShape)
        test = scalerY.transform(flat_test) if flat_test.shape[0]!=0 else test
        test = np.reshape(test, newShape)
        return train, val, test      

    def filtering_zeros(self,dataset):
        '''
        Eliminating sequences where all the time_steps are 0
        '''
        sumZero = np.sum(dataset, axis=1)
        idxNonZero = np.where(sumZero>0)
        return dataset[idxNonZero], idxNonZero

    def filtering_Off(self,dataset, building, newListApps):
        '''
        Eliminating sequences where all the time_steps are equal to value that can be considered OFF
        '''
        #assert len(self.listAppliances)==len(offValues[building])
        cond=[]
        for idx, app in enumerate(newListApps):
            #print(idx,building,app, dataset.shape)
            cond.append(dataset[:,:,idx]>offValues[building][app])#,dataset[:,1]>0]
        condArray = np.array(cond)
        print("cond array shape ",condArray.shape)
        goodRows = np.any(condArray,axis=0)
        #print("1st filter ", goodRows.shape)
        goodRows = np.any(goodRows,axis=1)
        #print("2nd filter ", goodRows.shape)
        idxNonZero = np.where(goodRows==True)
        print("len index ", len(idxNonZero), " shape ", len(idxNonZero[0]))
        return dataset[idxNonZero], idxNonZero

    def prepare_data(self, dataBuild, numApp, building, typeLoad):
        '''
        Spliting to scale over the training
        Filtering zeros before scaling
        Two ways:
            - when requiring just one appliance: spliting the disaggregated, 
                                                filtering zeros from disaggregated and then aggregated 
                                                scaling separetly
            - when requiring all the appliances: spliting the disaggregated,
                                                Calculating aggregated
                                                Filter zeros from aggregated and the disaggregated
                                                Scaling disaggregated
                                                Suming up aggregated from the scaled disaggregated
        '''
        indexVal, indexTest, indexEnd = self.calculate_split(dataBuild)
        print("Indexes: ",indexVal, indexTest, indexEnd)
        self.bin_edges = np.array([0,indexVal, indexTest, indexEnd])
        #Split and sum as if there was no specific appliance selected
        if (typeLoad==0):
            train_y, val_y, test_y = self.split_data(dataBuild, indexVal, indexTest, indexEnd, self.listAppliances)
        else:
            bigSet = self.convert3D(dataBuild,indexEnd)
            indexVal = int(self.pTrain * len(bigSet))
            indexTest = indexVal + int(self.pVal * len(bigSet))
            indexRandom =  np.random.permutation(len(bigSet))
            train_y = bigSet[indexRandom[:indexVal]]
            val_y = bigSet[indexRandom[indexVal:indexTest]]
            test_y = bigSet[indexRandom[indexTest:]]
        #sum up to calculate aggregation (x)
        train_x = np.sum(train_y, axis=2)
        val_x   = np.sum(val_y, axis=2)
        test_x  = np.sum(test_y, axis=2)

        print("Shapes before filtering ",train_x.shape, val_x.shape, test_x.shape, train_y.shape, val_y.shape, test_y.shape)
            
        if (numApp != -1):# if one specific appliance is selected
            #plt.figure(1)
            #plt.plot(dataBuild[self.listAppliances[numApp]][indexTest:])
            
            shapeY = [-1,self.time_steps]
            shapeX = [-1,self.time_steps]

            train_y, val_y, test_y = np.expand_dims(train_y[:,:,numApp],axis=2), np.expand_dims(val_y[:,:,numApp],axis=2),  np.expand_dims(test_y[:,:,numApp],axis=2)
            #Filtering zeros of specific appliance and respective aggregated instance

            if (self.flgFilterZeros==1):
                #train_y, idxTrain = self.filtering_zeros(train_y)
                #val_y, idxVal   = self.filtering_zeros(val_y)
                #test_y, idxTest  = self.filtering_zeros(test_y)

                train_y, idyTrain=self.filtering_Off(train_y, building,[self.listAppliances[numApp]])
                val_y, idyVal   = self.filtering_Off(val_y, building,[self.listAppliances[numApp]])
                test_y, idyTest  = self.filtering_Off(test_y, building,[self.listAppliances[numApp]])

                train_x = train_x[idyTrain]
                val_x   = val_x[idyVal]
                test_x  = test_x[idyTest]
            print("Shapes after filtering for one app ",train_x.shape, val_x.shape, test_x.shape, train_y.shape, val_y.shape, test_y.shape)

            #Scaling
            lenApps=1
            
            if (self.flgScaling==1):
                train_y, val_y, test_y = self.scaling(train_y, val_y, test_y, shapeY, thirdDim = lenApps)
                if (self.flgAggSumScaled==1):
                    pass # See the way to scale all x sets with respect to train_y
                train_x, val_x, test_x = self.scaling(train_x, val_x, test_x, shapeX, thirdDim = 1)
                
        else:
            lenApps = len(self.listAppliances)
            shapeY = [-1,self.time_steps, lenApps]
            shapeX = [-1,self.time_steps]

            #Filtering aggregated sequence with no information at all
            if (self.flgFilterZeros==1):
                '''
                train_x, idxTrain = self.filtering_zeros(train_x)
                val_x, idxVal   = self.filtering_zeros(val_x)
                test_x, idxTest  = self.filtering_zeros(test_x)
                '''
                train_y, idyTrain=self.filtering_Off(train_y, building,self.listAppliances)
                val_y, idyVal   = self.filtering_Off(val_y, building,self.listAppliances)
                test_y, idyTest  = self.filtering_Off(test_y, building,self.listAppliances)
                #Filtering the same instances in the disaggregated data set
                train_x = train_x[idyTrain]
                val_x   = val_x[idyVal]
                test_x  = test_x[idyTest]

            #Scaled each disaggregated separetly and recalculating aggregated consumption
            if (self.flgScaling==1):
                train_y, val_y, test_y = self.scaling(train_y, val_y, test_y, shapeY, thirdDim = lenApps)

                if (self.flgAggSumScaled ==1):
                    train_x = np.sum(train_y, axis=2)
                    val_x   = np.sum(val_y, axis=2)
                    test_x  = np.sum(test_y, axis=2)
                else:
                    train_x, val_x, test_x = self.scaling(train_x, val_x, test_x, shapeX, thirdDim = 1)
            print("Shapes after filtering all at once ",train_x.shape, val_x.shape, test_x.shape, train_y.shape, val_y.shape, test_y.shape)
        return [train_x, val_x, test_x, train_y, val_y, test_y]


    def load_csvdata(self, path, numApp, typeLoad=0):#appliances, filename, self.sample_period, windows
        '''
        Parameters:
            fileName
            numApp to indicate wether all the appliances should be read or just one of them

        Returns:
            totalX, totalY two dictionaries with the split of the X and Y in training, validation and testing
        '''        
        #homes = ["2859","3413","6990","7951","8292"] # collection of home id availables
        #home = homes[0]
        isMinutes = True
        lenApps = len(self.listAppliances)
        shapeY = [0,self.time_steps,lenApps] # (batch, seqLen, apps)
        
        if (numApp!=-1):
            lenApps = 1
            shapeY = [0,self.time_steps,lenApps]
        totalX = {'train':np.empty([0,self.time_steps]), 
                    'val':np.empty([0,self.time_steps]),
                    'test':np.empty([0,self.time_steps])}

        totalY = {'train':np.empty(shapeY), 
                    'val':np.empty(shapeY),
                    'test':np.empty(shapeY)}

        for building_i, window in self.windows.items():
            flgNewLoad = 0
            print(building_i)
            if (numApp!=-1):
                truFileName=str(building_i)+'_'+self.listAppliances[numApp]+'_'+str(self.time_steps)+'_'+str(self.stride_input)+'_'+window[0]+'_'+window[1]#fileName[pos:]
            else:
                truFileName=str(building_i)+'_'+'all'+'_'+str(self.time_steps)+'_'+str(self.stride_input)+'_'+window[0]+'_'+window[1]

            try:
                allSetsBuild = []*len(self.listAppliances)
                X = pickle.load( open(path+"/pickles/"+truFileName+"_X.pickle","rb"))
                Y = pickle.load( open(path+"/pickles/"+truFileName+"_Y.pickle","rb"))
                self.meanTrain = X['mean']
                self.stdTrain = X['std']
                allSetsBuild = X['train'], X['val'],X['test'], Y['train'], Y['val'], Y['test']
            except (OSError, IOError) as e:
                if(isMinutes):
                    imagepath=path+"/dataid_{}_minute.h5".format(building_i) 
                else:
                    imagepath=path+"/dataid_{}_hour.h5".format(building_i)
                #print(imagepath)
                imagearray  = pd.HDFStore(imagepath)
                #print imagearray.keys()
                data=imagearray['df']
                data = data.loc[data.index < window[1]]
                data = data.loc[data.index > window[0]]
                allSetsBuild = self.prepare_data(data, numApp,building_i, typeLoad)

                '''
                with open(path+"/pickles/"+truFileName+"_X.pickle",'wb') as fX:
                    pickle.dump({'mean':self.meanTrain,'std':self.stdTrain,'train':allSetsBuild[0],'val':allSetsBuild[1],'test':allSetsBuild[2]}, fX)
                with open(path+"/pickles/"+truFileName+"_Y.pickle",'wb') as fY:
                    pickle.dump({'train':allSetsBuild[3],'val':allSetsBuild[4],'test':allSetsBuild[5]},fY)
                '''
            print("One more building ", totalX['train'].shape, totalX['val'].shape, totalX['test'].shape, totalY['train'].shape, totalY['val'].shape, totalY['test'].shape)
            print(allSetsBuild[0].shape,allSetsBuild[3].shape)
            totalX['train'] = np.concatenate((totalX['train'], allSetsBuild[0]),axis=0)
            totalX['val'] = np.concatenate((totalX['val'], allSetsBuild[1]),axis=0)
            totalX['test'] = np.concatenate((totalX['test'], allSetsBuild[2]),axis=0)
            totalY['train'] = np.concatenate((totalY['train'], allSetsBuild[3]),axis=0)
            totalY['val'] = np.concatenate((totalY['val'], allSetsBuild[4]),axis=0)
            totalY['test'] = np.concatenate((totalY['test'], allSetsBuild[5]),axis=0)
            

            #os.makedirs(os.path.dirname(cwd+"/pickles/"+truFileName))
            #os.makedirs(os.path.dirname(cwd+"/pickles/"+truFileName))
            #this assumes you have a "pickles" directory at the same level as this file
        print(totalX['train'].shape, totalX['val'].shape, totalX['test'].shape, totalY['train'].shape, totalY['val'].shape, totalY['test'].shape)
        return totalX, totalY

    def build_dict_instances_plot(self,listDates, sizeBatch, TestSize):
      maxBatch = TestSize/sizeBatch  - 1
      listInst = []
      for strDate in listDates:#list range of dates per building. save some where the indexes of the total dataSet per building
        initialDate = datetime.datetime.strptime(self.windows[1][0], '%Y-%m-%d')
        targetDate = datetime.datetime.strptime(strDate, '%Y-%m-%d %H:%M')
        nInstance = (targetDate - initialDate).total_seconds()/6
        listInst.append(nInstance)

      instancesPlot = {}
      tupBinsOffset = []
      print(listInst)
      for inst in listInst:
        print(self.bin_edges)
        if (self.bin_edges[2]< inst and inst<self.bin_edges[3]): # If the instance is in the test set
          offSet = int((inst - self.bin_edges[2])/self.time_steps)
          tupBinsOffset.append((2,offSet))
      print(tupBinsOffset)
      dictInstances = {}
      for tupIns in tupBinsOffset:
        smallerIndexes = np.array(self.idxFiltered[tupIns[0]]) #Convert the indexs (tuples) in an array
        tupNumberSmaller = np.where(smallerIndexes<tupIns[1]) #Find the number of indexes that are smaller than the offset
        indexAfterFilter = len(tupNumberSmaller[0])
        nBatch = int(indexAfterFilter/sizeBatch) - 1
        indexAfterBatch =  (indexAfterFilter % sizeBatch) -1
        if (nBatch <= maxBatch):
          if nBatch in dictInstances:
            dictInstances[nBatch].append(int(indexAfterBatch))
          else:
            dictInstances[nBatch] = [int(indexAfterBatch)]
      print(dictInstances)
      return dictInstances