# USC RESEARCH TIME SERIES RMD NEURAL NETWORKS
import matplotlib
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import LSTM, Merge, Embedding, Activation, Dense, Convolution1D, Reshape, Dropout
import os
import argparse
import math
import time
from keras.callbacks import History
from keras.models import model_from_json

##############################################################################################################
#Script for utilizing different types of Neural Networks for predicting RMD parameters utilizing Bond Series info

#V1.0: 11/06/16 -> Setup with a basic 1D backpropagated network, Need to test with LSTM and CONV, no HPC
#V1.1: 11/16/16 -> Setup a basic LSTM with an embedding layer beforehand, very faulty, need to fix
#V1.2: 11/24/16 -> Added a merge layer so that my time sequences are managed and kept in original form when being fed
#V1.3: 11/30/16 -> Still working on LSTM, added user controls to switch between types of networks, epochs, graphs, and 
#V1.4: 12/01/16 -> Have LSTM up and running with new sequences and no embedding layer
#V1.5: 12/05/16 -> Have LSTM up and running with very promising results
#V1.6: 12/12/16 -> Have DNN up and running with very promising results


##############################################################################################################


def main():
    verbosity = 2
    if args.hpc == True:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = int(args.epochs)
    load = str(args.load)
    load = load.split(":")
    input_y = 401
    input_x = 1
    in_neurons = input_y*input_x 
    hidden_neurons = 401*3
    hidden_neurons2 = 600
    out_neurons = 3
    num_data_points = 105
    split = 0.7
    batch_size= 50    
    currentdir = os.getcwd()
    save_network_name = ""
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
        x_train4 = np.zeros((resized_num))
        for i,line in enumerate(x_train):
            x_train1[i] = line[:,1]
            x_train2[i] = line[:,2]
            x_train3[i] = line[:,3]
    #        x_train4[i] = line[:,3]
        resized_num2 = (len(x_test),input_y*input_x)
        x_test1 = np.zeros((resized_num2))
        x_test2 = np.zeros((resized_num2))
        x_test3 = np.zeros((resized_num2))
        x_test4 = np.zeros((resized_num2))
        for i,line in enumerate(x_test):
            x_test1[i] = line[:,1]
            x_test2[i] = line[:,2]
            x_test3[i] = line[:,3]
    #        x_test4[i] = line[:,3]
        return(x_train1,x_train2,x_train3,y_train,x_test1,x_test2,x_test3,y_test)    
        
    input_ts = create_dataset(num_data_points)
    x_data = input_ts[0]
    revised_x = np.zeros((num_data_points,401,4))
###################################### CHANGING TO VELOCITIES #########################################################
    for k,line in enumerate(x_data):
        for p,line2 in enumerate(line):
            if p == 0:
                continue
            revised_x[k][p] = p,line2[0]-x_data[k][p-1][0], line2[1]-x_data[k][p-1][1], line2[2]-x_data[k][p-1][2]
####################################################################################################################### 
    x_train1, x_train2, x_train3, y_train, x_test1, x_test2, x_test3, y_test = split_data_seq(revised_x,input_ts[1],split)

   
    if args.dnn:
        save_network_name = "DNN_Model"
        model = Sequential()
        
        left = Sequential()
        left.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
        middle = Sequential()
        middle.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
        right = Sequential()
        right.add(Dense(in_neurons,input_dim=in_neurons, activation="relu"))
        merged = Merge([left,middle,right],mode= 'concat',concat_axis=1)
        #merged2 = np.concatenate((x_train1,x_train2,x_train3))
        #print merged2
        model.add(merged)
        model.add(Dropout(0.2))
        model.add(Dense(hidden_neurons, activation="relu"))
        model.add(Dense(hidden_neurons2, activation="relu"))
        model.add(Dense(out_neurons))  
        model.compile(loss="mean_squared_error", optimizer="rmsprop")
        model.summary()
        history = model.fit([x_train1,x_train2,x_train3],y_train,validation_data=([x_test1,x_test2,x_test3],y_test),nb_epoch=epochs,verbose=verbosity)

        
    if args.lstm:
        save_network_name = "LSTM_Model"
        model = Sequential()
        
        left = Sequential()
        left.add(Dense(in_neurons,input_dim=in_neurons))
        #time included
        left2 = Sequential()
        left2.add(Dense(in_neurons,input_dim=in_neurons))
        
        middle = Sequential()
        middle.add(Dense(in_neurons,input_dim=in_neurons))
        right = Sequential()
        right.add(Dense(in_neurons,input_dim=in_neurons))
        
        #Time excluded
        merged = Merge([left,middle,right],mode= 'concat',concat_axis=1)
        #merged2 = np.concatenate((x_train1,x_train2,x_train3))
        #Time included
        #merged = Merge([left2,left,middle,right],mode= 'concat',concat_axis=1)
        #merged2 = np.concatenate((x_train1,x_train2,x_train3))
        
        model.add(merged)
        model.add(Reshape((1,hidden_neurons)))
        model.add(LSTM(401,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100,return_sequences=False))
        model.add(Dropout(0.2))
        #model.add(Dense(50))
        model.add(Dense(3))
        model.compile(loss='mse', optimizer='rmsprop')
        model.summary()
        history = model.fit([x_train1,x_train2,x_train3],y_train,validation_data=([x_test1,x_test2,x_test3],y_test),nb_epoch=epochs,verbose=verbosity)    
      
    if args.save != False:  
        # serialize model to JSON
        os.chdir(currentdir)
        model_json = model.to_json()
        showtime = time.strftime("%m_%d_%Y")
        dir_networks = str(showtime+"_networks")
        loc_dir = os.path.abspath(dir_networks)
        if os.path.isdir(loc_dir) == False:
            os.mkdir(dir_networks)
        os.chdir(dir_networks)
        save_network_name_j = str(save_network_name + ".json")
        save_network_name_h = str(save_network_name + ".h5")
        loc = os.path.abspath(save_network_name_j)
        i = 1
        while os.path.isfile(loc) == True:
            save_network_name_h = str(save_network_name + str(i) + ".h5")
            save_network_name_j = str(save_network_name + str(i) + ".json")
            loc = os.path.abspath(save_network_name_j)
        with open(save_network_name_j, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(save_network_name_h)
        print("Saved model to disk")
        
    if args.load != False:
        # load json and create model
        os.chdir(currentdir)
        json_file = open(str(load[0]), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(load[1])
        print("Loaded model from disk")

        loaded_model.compile(loss='mse', optimizer='rmsprop')
        
        testScore = loaded_model.evaluate([x_test1,x_test2,x_test3], y_test, verbose=2)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
        testPredict = loaded_model.predict([x_test1,x_test2,x_test3])
        
        total_y = np.concatenate((y_test,testPredict), axis=1)
        error = np.zeros(len(total_y))
        for i,line in enumerate(total_y):
            a = line[0] - line[3]
            b = line[1] - line[4]
            c = line[2] - line[5]
            tot = abs(a) + abs(b) + abs(c)
            error[i] = tot
        
        total = pandas.DataFrame(total_y)
        total['error'] = error
        total.rename(columns={0: 'actual_1', 1: 'actual_2', 2: 'actual_3', 3: 'predict_1', 4: 'predict_2', 5: 'predict_3'}, inplace=True)
        
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_title("Predicted test",fontsize=15)
        ax.plot(testPredict)
        ax.set_ylim([0.5,3.5])
        
        
        ax2 = fig.add_subplot(212)
        ax2.set_title("Actual test",fontsize=15)
        ax2.plot(y_test)
        ax2.set_ylim([0.5,3.5])
        
        print "Test Max absolute error: ", total[['error']].max(axis=0)[0]
        print "Test Mean absolute error: ", total[['error']].mean(axis=0)[0]
        
    if args.load == False:
        trainScore = model.evaluate([x_train1,x_train2,x_train3], y_train, verbose=2)
        print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
        trainPredict = model.predict([x_train1,x_train2,x_train3])
        
        testScore = model.evaluate([x_test1,x_test2,x_test3], y_test, verbose=2)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
        testPredict = model.predict([x_test1,x_test2,x_test3])
        
        
        total_y = np.concatenate((y_test,testPredict), axis=1)
        error = np.zeros(len(total_y))
        for i,line in enumerate(total_y):
            a = line[0] - line[3]
            b = line[1] - line[4]
            c = line[2] - line[5]
            tot = abs(a) + abs(b) + abs(c)
            error[i] = tot
        
        total = pandas.DataFrame(total_y)
        total['error'] = error
        total.rename(columns={0: 'actual_1', 1: 'actual_2', 2: 'actual_3', 3: 'predict_1', 4: 'predict_2', 5: 'predict_3'}, inplace=True)
        total.to_pickle("predict_actual_error_LSTM.pkl")
        
        fig2 = plt.figure()
        ax1 = fig2.add_subplot(111)
        ax1.set_title("loss vs epoch",fontsize=15)
        ax1.plot(history.history['loss'],label = 'Training')
        ax1.plot(history.history['val_loss'],label = 'Test')
        ax1.legend(loc='upper right', shadow=True)
        ax1.set_yscale('log')
        ax1.set_ylabel('Mean Squared Error', fontsize=15)
        ax1.set_xlabel('Epoch', fontsize=15)
    
        # PLOTTING RESULTS
#        testing_data_points = int(num_data_points-num_data_points*split)
        testing_data_points = 20
#        training_data_points = int(num_data_points*split)
        training_data_points = 50
        
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
        showtime = time.strftime("%m_%d_%Y")
        dir_graphs = str(showtime+"_graphs")
        location_dir = os.path.abspath(dir_graphs)
        if os.path.isdir(location_dir) == False:
            os.mkdir(dir_graphs)
        os.chdir(dir_graphs)
        saveline = str("Results_" + type_network + "_" + showtime + ".png")
        saveline2 = str("Results_epochs_MSE_" + type_network + "_" + showtime + ".png")
        location = os.path.abspath(saveline)
        i = 1
        while os.path.isfile(location) == True:
            saveline = str(str("Results_" + type_network + "_" + showtime) + "_" + str(i) + ".png")
            saveline2 = str(str("Results_epochs_MSE_" + type_network + "_" + showtime) + "_" + str(i) + ".png")
            location = os.path.abspath(saveline)
            i += 1
        fig.savefig(saveline)
        if args.load == False:
            fig2.savefig(saveline2)
        os.chdir("..")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Script for utilizing different types of Neural Networks for predicting RMD parameters utilizing Bond Series info')
    parser.add_argument('-dnn', help='Using the deep neural network (not CONV or LSTM) basic 2D', default=False, action='store_true',dest="dnn")
    parser.add_argument('-lstm', help='Using the Recurrent network, and in particular the LSTM option', default=False, action='store_true',dest="lstm")
    parser.add_argument('-graphs', help='Save a graph of predictions with their actual values for the training and test datasets', default=False, action='store_true',dest="graphs")
    parser.add_argument('-epochs', help='How many itterations do you want to run your Network with?  (default = 25)',default=25, action='store', dest = 'epochs')
    parser.add_argument('-hpc', help='This is used so that you can run this on the HPC cluster, still need to specify graphs if you want to save figure', default=False, action='store_true',dest='hpc')
    parser.add_argument('-load', help='This to load a previously generated and trained network from a json and h5 file, just put two file names, json first then h5 file (within the current directory) seperated by a :, NO SPACES, i.e. ex1.json:ex1.h5',default=False,action='store',dest='load')
    parser.add_argument('-save', help='This is used to save the networks weights and structure, after its trained', default=False, action='store_true',dest='save')
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
    if args.load != False:
        type_network = "pre_loaded"
        print "\n", startline, type_network, endline, "\n", graphs_display, "\n"
    if args.save == True:
        print "\n", "After training your network will be saved in the date_networks directory", "\n"
    if args.dnn == False and args.lstm == False and args.load == False:
        print "You haven't selected a Neural Network to use or there is a problem with your input. Check the help!", "\n", "\n"
    main()
