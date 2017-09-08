To run this program, some parameters are needed:

python lstm_tf.py filePath sequenceLength hiddenUnits numEpochs strideTraining strideTest

Where:
filePath: the path of the input data

sequenceLength: he input file is a unique large sequence of values. To build a batch, we generate several sequences of this length to become the training or test set.

hiddnUnits: the number of units in the first hidden layer of the lstm

numEpochs: number of epochs to consider in the training

strideTraining: the larger the strideTraining the less number of sequences in the training Set. If this is smaller than the sequenceLength, the sequences for the 
training are overlapped.

strideTest: if we test with overlapped sequences, we will have to average the output for certain points as they do it in Kelly's paper. Other option is to set the 

strideTest equal to the sequenceLength and to get just one prediction per timeStep (from one sequence).
