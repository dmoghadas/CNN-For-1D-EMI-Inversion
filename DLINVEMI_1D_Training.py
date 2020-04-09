### DLINVEMI_1D_Training: program to train 1D CNN for EMI inversion
### Written by Davood Moghadas
### Brandenburg University of Technology (BTU), Cottbus-Senftenberg
### Version: 23.09.2019
###
### libraries
import numpy as np
import pandas as pd
from tensorflow import keras
from numpy.random import seed
from tensorflow import set_random_seed
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

### load and get data
data = np.load('Fdata.npz', allow_pickle=True)
Headers = list(data.keys())
for i in range(0, len(Headers)):
    H = Headers[i]
    str2 = H + ' = data[' + '"' + H + '"' + ']'
    exec(str2)

### make input and output
ECa_hcp = ECa_hcp.T
ECa_vcp = ECa_vcp.T
EC = RL_lin.T
ECa = np.append(ECa_vcp, ECa_hcp, axis=1)
input = ECa.copy()
output = EC.copy()

### function to split data
def data_split_seq(data, N, pr_train, pr_test):
    ### N :  total number of data points
    pr_train = pr_train/100
    pr_test = pr_test/100
    N_train = int(N * pr_train)
    N_test = int(N * pr_test)
    index_train = np.arange(0, N_train,1)
    index_test = np.arange(0, N_test, 1)
    train = data[index_train, :]
    test1 = np.delete(data, index_train, axis=0)
    test = test1[index_test, :]
    validate = np.delete(test1, index_test, axis=0)
    return train, validate, test

## function to reshape input data
def input_prep(X, n_features):
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X

## Split data into training validation and test subsets
input_train, input_val, input_test = data_split_seq(input,input.shape[0],70,15)
output_train, output_val, output_test = data_split_seq(output,output.shape[0],70,15)

## data normalization
scaler_in = MinMaxScaler(feature_range=(0, 1))
scaler_out = MinMaxScaler(feature_range=(0, 1))
scaler_in = scaler_in.fit(input)
scaler_out = scaler_out.fit(output)

input_train_norm = scaler_in.transform(input_train)
input_val_norm = scaler_in.transform(input_val)
input_test_norm = scaler_in.transform(input_test)
output_train_norm = scaler_out.transform(output_train)
output_val_norm = scaler_out.transform(output_val)
output_test_norm = scaler_out.transform(output_test)

### number of input and outputs and features
n_features = 1
n_input, n_output = input.shape[1], output.shape[1]

### make input
input_train_norm = input_prep(input_train_norm, n_features)
input_val_norm = input_prep(input_val_norm, n_features)
input_test_norm = input_prep(input_test_norm, n_features)

## Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

### set seed
## for keras
seed(0)
## for tensorflow
set_random_seed(0)

### define model
model = keras.Sequential()

## leakage of the leaky relu
LRU = 0.01

### the first set of convolutional layers
model.add(keras.layers.Conv1D(filters=6, kernel_size=2, strides=1, padding="same", input_shape=(n_input, n_features)))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(keras.layers.Conv1D(filters=12, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(keras.layers.Conv1D(filters=24, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(keras.layers.Conv1D(filters=48, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))

### the second set of convolutional layers
model.add(keras.layers.Conv1D(filters=24, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.Dropout(0.02))
model.add(keras.layers.Conv1D(filters=12, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.Dropout(0.01))
model.add(keras.layers.Conv1D(filters=6, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))

model.add(keras.layers.Flatten())

### print layers
print(model.summary())

### training setting
opt = keras.optimizers.Adam(lr=0.00005)
str_metrics = ['mean_squared_error', 'mean_absolute_error']
str_labels = ['MSE', 'MAE']
loss = 'mse'

### train CNN
model.compile(optimizer=opt, loss=loss, metrics=str_metrics)

### early stop
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(input_train_norm, output_train_norm, epochs=1000,
                    validation_data = (input_val_norm, output_val_norm),
                    verbose=0, callbacks=[early_stop, PrintDot()])

### training history
tr_history = pd.DataFrame(history.history)
tr_history['epoch'] = history.epoch

## Make predictions
train_predictions_norm = model.predict(input_train_norm)
val_predictions_norm = model.predict(input_val_norm)
test_predictions_norm = model.predict(input_test_norm)
train_predictions = scaler_out.inverse_transform(train_predictions_norm)
val_predictions = scaler_out.inverse_transform(val_predictions_norm)
test_predictions = scaler_out.inverse_transform(test_predictions_norm)

## make output
cnn_data = {'input_train': input_train, 'input_test': input_test,'input_val': input_val,
            'output_train': output_train, 'output_test': output_test, 'output_val': output_val,
            'test_predictions':test_predictions,'val_predictions':val_predictions,'train_predictions':train_predictions,
            'depth': depth, 'depth_lin': depth_lin, 'str_metrics': str_metrics, 'str_labels': str_labels}

## save data
joblib.dump(scaler_in, 'scaler_in.pkl')
joblib.dump(scaler_out, 'scaler_out.pkl')
np.savez_compressed('cnn_data',**cnn_data)
model.save('cnn_model.h5')
with open('tr_history.pickle', 'wb') as f:
    pickle.dump(tr_history, f)

### convert to numpy array
M1_train = np.array(output_train)
M1_val = np.array(output_val)
M1_test = np.array(output_test)
M2_train = np.array(train_predictions)
M2_val = np.array(val_predictions)
M2_test = np.array(test_predictions)

### evaluations
### function to plot history
def plot_history(hist, str_metrics, str_labels):
    n = len(str_metrics)
    if n > 1:
        fig, ax = plt.subplots(n, 1)
        for ii in range(0, n):
            str_tr = str_metrics[ii]
            str_val = 'val_' + str_metrics[ii]
            str_y = str_labels[ii]
            ax[ii].plot(hist['epoch'], hist[str_tr], '-b', label='Training Error', linewidth=2)
            ax[ii].plot(hist['epoch'], hist[str_val], '-r', label='Validation Error', linewidth=2)
            ax[ii].grid()
            ax[ii].set_xlabel('Epoch')
            ax[ii].set_ylabel(str_y)
            ax[ii].legend()
    else:
        fig = plt.figure()
        ax = plt.gca()
        str_tr = str_metrics[0]
        str_val = 'val_' + str_metrics[0]
        str_y = str_labels[0]

        ax.plot(hist['epoch'], hist[str_tr], '-b', label='Training Error', linewidth=2)
        ax.plot(hist['epoch'], hist[str_val], '-r', label='Validation Error', linewidth=2)
        ax.grid()
        ax.set_xlabel('Epoch')
        ax.set_ylabel(str_y)
        ax.legend()
    plt.show()

### function to plot error histogram
def plot_errorhist(nbins, M1_train, M2_train, M1_val, M2_val):
    error_train = M2_train.ravel() - M1_train.ravel()
    error_val = M2_val.ravel() - M1_val.ravel()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.hist(error_train, bins = nbins, color='b')
    ax1.set_title('Training')
    ax1.set_ylabel('Frequency')
    ax2.hist(error_val, bins = nbins, color='b')
    ax2.set_title('Validation')
    ax2.set_ylabel('Frequency')
    plt.show()

### plot history
plot_history(tr_history, str_metrics, str_labels)
### plot error residuals
plot_errorhist(50, M1_train, M2_train, M1_val, M2_val)
