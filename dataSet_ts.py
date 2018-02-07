import numpy as np
import pandas as pd
import nilmtk
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing


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
        indexVal = int(round((self.pTrain*self.stride_input*(completeInstances - self.time_steps) +  
                              pRest*self.stride_output*self.time_steps)/ 
                                (pRest*self.stride_output + self.pTrain*self.stride_input)))
        indexTest = int(indexVal + self.pVal*(completeInstances - indexVal)/pRest)

        #Sabina 50% - 25% - 25%
        return indexVal,indexTest,completeInstances
    
    def split_data(self, dataBuild, indexVal, indexTest, indexEnd,apps):
        train_y = self.rnn_data(dataBuild, 0, indexVal, apps, self.stride_input, labels=True)
        val_y   = self.rnn_data(dataBuild, indexVal, indexTest, apps, self.stride_output, labels=True)
        test_y  = self.rnn_data(dataBuild, indexTest, indexEnd, apps, self.stride_output, labels=True)
        return train_y, val_y, test_y

    def scaling(self, train, val, test, newShape, thirdDim):
            ##### scaling
        flat_train = np.reshape(train,[-1,thirdDim])
        flat_val = np.reshape(val,[-1,thirdDim])
        flat_test = np.reshape(test,[-1,thirdDim])

        scalerY = preprocessing.StandardScaler().fit(flat_train)

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

    def prepare_data(self, dataBuild, numApp):
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
        
        #Split and sum as if there was no specific appliance selected
        train_y, val_y, test_y = self.split_data(dataBuild, indexVal, indexTest, indexEnd, self.listAppliances)

        #sum up to calculate aggregation (x)
        train_x = np.sum(train_y, axis=2)
        val_x   = np.sum(val_y, axis=2)
        test_x  = np.sum(test_y, axis=2)

        print(train_x.shape, val_x.shape, test_x.shape, train_y.shape, val_y.shape, test_y.shape)
            
        if (numApp != -1):# if one specific appliance is selected
            #plt.figure(1)
            #plt.plot(dataBuild[self.listAppliances[numApp]][indexTest:])
            
            shapeY = [-1,self.time_steps]
            shapeX = [-1,self.time_steps]

            train_y, val_y, test_y = train_y[:,:,numApp], val_y[:,:,numApp], test_y[:,:,numApp]
            
            #Filtering zeros of specific appliance and respective aggregated instance
            if (self.flgFilterZeros==1):
                train_y, idxTrain = self.filtering_zeros(train_y)
                val_y, idxVal   = self.filtering_zeros(val_y)
                test_y, idxTest  = self.filtering_zeros(test_y)

                train_x = train_x[idxTrain]
                val_x   = val_x[idxVal]
                test_x  = test_x[idxTest]

            #Scaling
            lenApps=1
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
                train_x, idxTrain = self.filtering_zeros(train_x)
                val_x, idxVal   = self.filtering_zeros(val_x)
                test_x, idxTest  = self.filtering_zeros(test_x)

                #Filtering the same instances in the disaggregated data set
                train_y = train_y[idxTrain]
                val_y   = val_y[idxVal]
                test_y  = test_y[idxTest]

            #Scaled each disaggregated separetly and recalculating aggregated consumption
            train_y, val_y, test_y = self.scaling(train_y, val_y, test_y, shapeY, thirdDim = lenApps)

            if (self.flgAggSumScaled ==1):
                train_x = np.sum(train_y, axis=2)
                val_x   = np.sum(val_y, axis=2)
                test_x  = np.sum(test_y, axis=2)
            else:
                train_x, val_x, test_x = self.scaling(train_x, val_x, test_x, shapeX, thirdDim = 1)

        return [train_x, val_x, test_x, train_y, val_y, test_y]

    def all_building_data(self,dataset,building, window):
        dfTotal = []
        df_build = pd.DataFrame(columns=['building']+self.listAppliances+['use'])#
        dataset.set_window(*window)
        elec = dataset.buildings[building].elec
        for appliance in self.listAppliances:
            pdAppSeries = pd.Series()
            # Because more than one seems not to be the total number of instances for that window
            serieApp = elec[appliance].power_series(sample_period=self.sample_period, resample=True).next() #pandas.core.series.Series
            if (len(serieApp.index) !=0): # Some buildings may not have all the appliances
                pdAppSeries = pdAppSeries.append(serieApp)
            df_build[appliance] =  pdAppSeries
        sizeBuild = len(df_build.index)
        df_build['building'] = [building for row in range(sizeBuild)]
        df_build['use'] = df_build[self.listAppliances].sum(axis=1)
        return df_build.fillna(0)


    def load_csvdata(self, fileName, numApp):#appliances, filename, self.sample_period, windows
        '''
        Parameters:
            fileName
            numApp to indicate wether all the appliances should be read or just one of them

        Returns:
            totalX, totalY two dictionaries with the split of the X and Y in training, validation and testing
        '''        
        pos = fileName.rfind("/")+1
        cwd=os.getcwd()
        isDataPort = True
        homes = ["2859","3413","6990","7951","8292"] # collection of home id availables
        home = homes[0]
        isMinutes = True
        path = fileName[:pos]
        lenApps = len(self.listAppliances)
        shapeY = [0,self.time_steps,lenApps] # (batch, seqLen, apps)
        
        if (numApp!=-1):
            lenApps = 1
            shapeY = [0,self.time_steps]
        totalX = {'train':np.empty([0,self.time_steps]), 
                    'val':np.empty([0,self.time_steps]),
                    'test':np.empty([0,self.time_steps])}

        totalY = {'train':np.empty(shapeY), 
                    'val':np.empty(shapeY),
                    'test':np.empty(shapeY)}

        for building_i, window in self.windows.items():
            flgNewLoad = 0
            try:
                if (numApp!=-1):
                    truFileName=str(building_i)+'_'+self.listAppliances[numApp]+'_'+'_'+window[0]+'_'+window[1]#fileName[pos:]
                else:
                    truFileName=str(building_i)+'_'+'all'+'_'+'_'+window[0]+'_'+window[1]
                allSetsBuild = []*6
                X = pickle.load( open(path+"/pickles/"+truFileName+"_X.pickle","rb"))
                Y = pickle.load( open(path+"/pickles/"+truFileName+"_Y.pickle","rb"))
                allSetsBuild = X['train'], X['val'],X['test'], Y['train'], Y['val'], Y['test']
            except (OSError, IOError) as e:
                if(isDataPort == False):
                    dataset = nilmtk.DataSet(fileName)
                    flgNewLoad = 1
                    dataBuild = self.all_building_data(dataset,building_i, window)
                    allSetsBuild = self.prepare_data(dataBuild, numApp)
                else:
                    self.listAppliances = ['air1', 'furnace1', 'refrigerator1',  'clotheswasher1','drye1','dishwasher1', 'kitchenapp1', 'microwave1'] #TEMPORARY MEASURE! REMOVE ASAP
                    if(isMinutes):
                        imagepath=path+"dataid_{}_minute.h5".format(building_i) 
                    else:
                        imagepath=path+"dataid_{}_hour.h5".format(building_i)
	            imagearray  = pd.HDFStore(imagepath)
	            #print imagearray.keys()
	            data=imagearray['df']
                    data = data.loc[data.index < window[1]]
                    data = data.loc[data.index>window[0]]
   
                    allSetsBuild = self.prepare_data(data, numApp)
            totalX['train'] = np.concatenate((totalX['train'], allSetsBuild[0]),axis=0)
            totalX['val'] = np.concatenate((totalX['val'], allSetsBuild[1]),axis=0)
            totalX['test'] = np.concatenate((totalX['test'], allSetsBuild[2]),axis=0)
            totalY['train'] = np.concatenate((totalY['train'], allSetsBuild[3]),axis=0)
            totalY['val'] = np.concatenate((totalY['val'], allSetsBuild[4]),axis=0)
            totalY['test'] = np.concatenate((totalY['test'], allSetsBuild[5]),axis=0)
            print(totalX['train'].shape, totalX['val'].shape, totalX['test'].shape, totalY['train'].shape, totalY['val'].shape, totalY['test'].shape)

            if (flgNewLoad==1):
                with open(path+"/pickles/"+truFileName+"_X.pickle",'wb') as fX:
                    pickle.dump(totalX, fX)
                with open(path+"/pickles/"+truFileName+"_Y.pickle",'wb') as fY:
                    pickle.dump(totalY,fY)
            #os.makedirs(os.path.dirname(cwd+"/pickles/"+truFileName))
            #os.makedirs(os.path.dirname(cwd+"/pickles/"+truFileName))
            #this assumes you have a "pickles" directory at the same level as this file
        print(totalX['train'].shape, totalX['val'].shape, totalX['test'].shape, totalY['train'].shape, totalY['val'].shape, totalY['test'].shape)
        print(data.dtypes)
        return totalX, totalY
