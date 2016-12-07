# USC RESEARCH TIME SERIES RMD NEURAL NETWORKS
import matplotlib
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import LSTM, Merge, Embedding, Activation, Dense, Convolution1D
import os
import argparse
import math
import time

##############################################################################################################
#Script for utilizing different types of Neural Networks for predicting RMD parameters utilizing Bond Series info

#V1.0: 11/06/16 -> Setup with a basic 1D backpropagated network, Need to test with LSTM and CONV, no HPC
#V1.1: 11/16/16 -> Setup a basic LSTM with an embedding layer beforehand, very faulty, need to fix
#V1.2: 11/24/16 -> Added a merge layer so that my time sequences are managed and kept in original form when being fed
#V1.3: 11/30/16 -> Still working on LSTM, added user controls to switch between types of networks, epochs, graphs, and 
#V1.4: 12/01/16 -> Have LSTM up and running with new sequences and no embedding layer



##############################################################################################################


def main():
    verbosity = 2
    abs_graphs = True
    if args.hpc == True:
        matplotlib.use('Agg')
        verbosity = 0
        abs_graphs = False
    import matplotlib.pyplot as plt
    epochs = int(args.epochs)
    input_y = 401
    input_x = 1
    in_neurons = input_y*input_x 
    hidden_neurons = 600
    hidden_neurons2 = 150
    out_neurons = 3
    num_data_points = 105
    split = 0.7
    batch_size= 50
    
    currentdir = os.getcwd()
    #Go to directory of data
    def create_dataset(num_data_points):
        os.chdir("RMD_timeseries/Data")
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
        
    input_ts = create_dataset(num_data_points)
    x_train1, x_train2, x_train3, y_train, x_test1, x_test2, x_test3, y_test = split_data_seq(input_ts[0],input_ts[1],split)
    
    if args.dnn:
        model = Sequential()
        
        left = Sequential()
        left.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
        middle = Sequential()
        middle.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
        right = Sequential()
        right.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
        merged = Merge([left,middle,right],mode= 'concat')
        model.add(merged)
        model.add(Dense(hidden_neurons, activation="relu"))  
        model.add(Dense(hidden_neurons2, activation="relu"))
        model.add(Dense(out_neurons))  
        model.compile(loss="mean_squared_error", optimizer="rmsprop")
        print model.summary()
        model.fit([x_train1,x_train2,x_train3],y_train,nb_epoch=epochs,verbose=verbosity)
        
    if args.lstm:
    #     CREATING THE FULLY CONNECTED LSTM ANN MODEL  
        model = Sequential()
        
        left = Sequential()
        left.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
        middle = Sequential()
        middle.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
        right = Sequential()
        right.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
        merged = Merge([left,middle,right],mode= 'concat')
        model.add(merged)
        #model.summary()
        model.add(LSTM(401,input_shape=(x_train1.shape[0],x_train2.shape[1])))
        model.add(Dense(150))
        model.add(Dense(30))
        model.add(Dense(3))
        model.compile(loss='mse', optimizer='rmsprop')
        model.summary()
        model.fit([x_train1,x_train2,x_train3],y_train,nb_epoch=epochs,verbose=verbosity)
    
    
    
    #Evaulation of contstructed network
    trainScore = model.evaluate([x_train1,x_train2,x_train3], y_train, verbose=2)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    trainPredict = model.predict([x_train1,x_train2,x_train3])
    
    testScore = model.evaluate([x_test1,x_test2,x_test3], y_test, verbose=2)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    testPredict = model.predict([x_test1,x_test2,x_test3])
    
    
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
    
    if args.graphs:
        os.chdir(currentdir)
        showtime = time.strftime("%d_%m_%Y")
        saveline = str("Results_" + type_network + "_" + showtime + ".png")
        location = os.path.abspath(saveline)
        i = 1
        while os.path.isfile(location) == True:
            saveline = str(str("Results_" + type_network + "_" + showtime) + "_" + str(i) + ".png")
            location = os.path.abspath(saveline)
            i += 1
        fig.savefig(saveline)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Script for utilizing different types of Neural Networks for predicting RMD parameters utilizing Bond Series info')
    parser.add_argument('-dnn', help='Using the deep neural network (not CONV or LSTM) basic 2D', default=False, action='store_true',dest="dnn")
    parser.add_argument('-lstm', help='Using the Recurrent network, and in particular the LSTM option', default=False, action='store_true',dest="lstm")
    parser.add_argument('-graphs', help='Save a graph of predictions with their actual values for the training and test datasets', default=False, action='store_true',dest="graphs")
    parser.add_argument('-epochs', help='How many itterations do you want to run your Network with?  (default = 200)',default=25, action='store', dest = 'epochs')
    parser.add_argument('-hpc', help='This is used so that you can run this on the HPC cluster, still need to specify graphs if you want to save figure', default=False, action='store_true',dest='hpc')
    args = parser.parse_args()
    
    startline = "You will be running the"
    endline = "Neural Network"
    graphs_display = "NO GRAPHS"
    if args.graphs == True:
        graphs_display = "GRAPHS ON"
    if args.lstm == True:
        type_network = "LSTM"
        print "\n", startline, type_network, endline, "\n", graphs_display, "\n"
    if args.dnn == True:
        type_network = "Deep_2D"
        print "\n", startline, type_network, endline, "\n", graphs_display, "\n"
    if args.dnn == False and args.lstm == False:
        print "You haven't selected a Neural Network to use or there is a problem with your input. Check the help!", "\n", "\n"
    main()
    
    
    
    #### Plot of points
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_title("time series of the C-C, C-O, Si-O bond formations")
    #ax.set_xlabel("time step")
    #ax.set_ylabel("# of bonds") 
    #ax.plot(ts,os, label = "C-O bonds")
    #ax.plot(ts,cs, label = "C-C bonds")
    #ax.plot(ts,sis, label = "Si-O bonds")
    #plt.legend(loc = 2)
    #plt.show()