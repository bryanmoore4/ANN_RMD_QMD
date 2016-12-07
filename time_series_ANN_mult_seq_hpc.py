# USC RESEARCH TIME SERIES RMD NEURAL NETWORKS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import LSTM, Merge, Embedding, Activation, Dense, Convolution1D, Reshape, Dropout
import os
import math


##############################################################################################################
#Script for utilizing different types of Neural Networks for predicting RMD parameters utilizing Bond Series info

#V1.0: 11/06/16 -> Setup with a basic 1D backpropagated network, Need to test with LSTM and CONV, no HPC
#V1.1: 11/16/16 -> Setup a basic LSTM with an embedding layer beforehand, very faulty, need to fix
#V1.2: 11/24/16 -> Added a merge layer so that my time sequences are managed and kept in original form when being fed
#V1.3: 11/30/16 -> Still working on LSTM, added user controls to switch between types of networks, epochs, graphs, and 
#V1.4: 12/01/16 -> Have LSTM up and running with new sequences and no embedding layer



##############################################################################################################



#Go to directory of data
def create_dataset(num_data_points):
    os.chdir("..")
    os.chdir("RMD_Training_ANN/data_and_results")
    x_data_array = np.zeros((num_data_points,401,3))
    y_data_array = np.zeros((num_data_points,3))
#    print(x_data_array)
    #Convert the data with pandas
    dataframe = pandas.read_csv("RMD_data_files.txt",delimiter = ":",engine = 'python',header=None) 
    dataframe = dataframe.values
    X_names = dataframe[:,0]
    X_names = X_names.astype('string')
    for i in range(num_data_points):
        X_name = X_names[i][-17:]
        X_frame = pandas.read_csv(X_name,delim_whitespace= True, usecols = [1,2,3],engine = 'python', header = None)
        X = X_frame.values
        X = X.astype('float32')
        x_data_array[i] = X
        Y = dataframe[:,1:4]
        Y = Y.astype('float32')
        Y = Y[i]
        y_data_array[i] = Y
    
    return(x_data_array,y_data_array)

def split_data(x_array,y_array,split):
    total_length = len(x_array)
    split_num = int(split*total_length)
    x_train1 = x_array[0:split_num]
    y_train = y_array[0:split_num]
    x_test1 = x_array[split_num+1:total_length]
    y_test = y_array[split_num+1:total_length]

    resized_num = (len(x_train1),input_y*input_x)
    x_train = np.zeros(resized_num)
    for i,line in enumerate(x_train1):
        x_train[i] = np.ndarray.flatten(line)
    resized_num2 = (len(x_test1),input_y*input_x)
    x_test = np.zeros(resized_num2)
    for i,line in enumerate(x_test1):
        x_test[i] = np.ndarray.flatten(line)
        
    return(x_train,y_train,x_test,y_test)
    
def split_data_seq(x_array,y_array,split):
    total_length = len(x_array)
    split_num = int(split*total_length)
    x_train = x_array[0:split_num]
    y_train = y_array[0:split_num]
    x_test = x_array[split_num+1:total_length]
    y_test = y_array[split_num+1:total_length]
    resized_num = (len(x_train),input_y*input_x)
    x_train1 = np.zeros((resized_num))
    x_train2 = np.zeros((resized_num))
    x_train3 = np.zeros((resized_num))
    for i,line in enumerate(x_train):
        x_train1[i] = line[:,0]
        x_train2[i] = line[:,1]
        x_train3[i] = line[:,2]
    resized_num2 = (len(x_test),input_y*input_x)
    x_test1 = np.zeros((resized_num2))
    x_test2 = np.zeros((resized_num2))
    x_test3 = np.zeros((resized_num2))
    for i,line in enumerate(x_test):
        x_test1[i] = line[:,0]
        x_test2[i] = line[:,1]
        x_test3[i] = line[:,2]
        
    return(x_train1,x_train2,x_train3,y_train,x_test1,x_test2,x_test3,y_test)
    
input_y = 401
input_x = 1
in_neurons = input_y*input_x 
hidden_neurons = 1203
hidden_neurons2 = 600
out_neurons = 3
num_data_points = 105
split = 0.7
input_ts = create_dataset(num_data_points)
batch_size= 50

x_train1, x_train2, x_train3, y_train, x_test1, x_test2, x_test3, y_test = split_data_seq(input_ts[0],input_ts[1],split)

#################################################################################################
#CREATING THE NETWORK
#################################################################################################

#model = Sequential()
#
#left = Sequential()
#left.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
#middle = Sequential()
#middle.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
#right = Sequential()
#right.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
#merged = Merge([left,middle,right],mode= 'concat',concat_axis=1)
##merged2 = np.concatenate((x_train1,x_train2,x_train3))
##print merged2
#model.add(merged)
#model.add(Dense(hidden_neurons, activation="relu"))  
#model.add(Dense(hidden_neurons2, activation="relu"))
#model.add(Dense(out_neurons))  
#model.compile(loss="mean_squared_error", optimizer="rmsprop")
#print model.summary()
#model.fit([x_train1,x_train2,x_train3],y_train,nb_epoch=200,verbose=2)


# CREATING THE FULLY CONNECTED LSTM ANN MODEL
model = Sequential()

left = Sequential()
left.add(Dense(in_neurons,input_dim=in_neurons))
middle = Sequential()
middle.add(Dense(in_neurons,input_dim=in_neurons))
right = Sequential()
right.add(Dense(in_neurons,input_dim=in_neurons))
merged = Merge([left,middle,right],mode= 'concat',concat_axis=1)
merged2 = np.concatenate((x_train1,x_train2,x_train3))
model.add(merged)
model.add(Reshape((1,hidden_neurons)))
model.add(LSTM(800,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(250,return_sequences=False))
model.add(Dropout(0.2))
#model.add(Dense(50))
model.add(Dense(3))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()
model.fit([x_train1,x_train2,x_train3],y_train,nb_epoch=50,verbose=2)



#Evaulation of contstructed network
trainScore = model.evaluate([x_train1,x_train2,x_train3], y_train, verbose=2)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
trainPredict = model.predict([x_train1,x_train2,x_train3])

testScore = model.evaluate([x_test1,x_test2,x_test3], y_test, verbose=2)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
testPredict = model.predict([x_test1,x_test2,x_test3])


os.chdir("..")
os.chdir("predict_Bryan")
# PLOTTING RESULTS
testing_data_points = int(num_data_points-num_data_points*split)
training_data_points = int(num_data_points*split)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.set_title("Predicted test",fontsize=15)
ax.plot(testPredict)
ax.set_ylim([0.5,3.5])
ax.set_xlim(0,testing_data_points+int(testing_data_points*0.05))

ax2 = fig.add_subplot(223)
ax2.set_title("Actual test",fontsize=15)
ax2.plot(y_test)
ax2.set_ylim([0.5,3.5])
ax2.set_xlim(0,testing_data_points+int(testing_data_points*0.05))

ax3 = fig.add_subplot(222)
ax3.set_title("Predicted train",fontsize=15)
ax3.plot(trainPredict)
ax3.set_ylim([0.5,3.5])
ax3.set_xlim(0,training_data_points+int(training_data_points*0.05))

ax4 = fig.add_subplot(224)
ax4.set_title("Actual train",fontsize=15)
ax4.plot(y_train)
ax4.set_ylim([0.5,3.5])
ax4.set_xlim(0,training_data_points+int(training_data_points*0.05))

fig.savefig("LSTMs_Network_test.png")