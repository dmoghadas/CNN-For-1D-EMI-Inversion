### DLINVEMI_1D_Predictions: program to estimate subsurface EC distribution from EMI data using DL inversion
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from scipy.interpolate import interp1d

### load and get data
scaler_in = joblib.load('scaler_in.pkl')
scaler_out = joblib.load('scaler_out.pkl')
data = np.load('cnn_data.npz', allow_pickle=True)
model = keras.models.load_model('cnn_model.h5')
Fdata = np.load('Fdata.npz', allow_pickle=True)
EMI = np.load('EMI.npz', allow_pickle=True)

data = dict(data)
Headers = list(data.keys())
for i in range(0, len(Headers)):
    H = Headers[i]
    str2 = H + ' = data[' + '"' + H + '"' + ']'
    exec(str2)

ECa_hcp = EMI['ECa_hcp']
ECa_vcp = EMI['ECa_vcp']
X = EMI['X']

## function to reshape input data
def input_prep(X, n_features):
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X

### function to predict EC data using CNN model
def cnn_predict(ECa_hcp, ECa_vcp, model, scaler_in, scaler_out, depth, depth_lin):
    ECa_hcp = ECa_hcp.T
    ECa_vcp = ECa_vcp.T
    ECa = np.append(ECa_vcp, ECa_hcp, axis=1)
    ECa_norm = scaler_in.transform(ECa)
    n_features = 1
    ECa_norm = input_prep(ECa_norm, n_features)
    EC_norm = model.predict(ECa_norm)
    EC = scaler_out.inverse_transform(EC_norm)
    EC = EC.T
    ### interpolate to the linear depth
    EC2 = np.zeros(np.shape(EC))
    for i in range(0, EC.shape[1]):
        ECC = EC[:, i].ravel()
        depth_lin2 = np.append(0, depth_lin.ravel())
        ECC2 = np.append(ECC[0], ECC)
        f = interp1d(depth_lin2.ravel(), ECC2.ravel())
        EC2[:, i] = f(depth.ravel())
    EC = EC2.copy()
    return EC

### make predictions
EC = cnn_predict(ECa_hcp, ECa_vcp, model, scaler_in, scaler_out, depth, depth_lin)

### plot data
x1 = X.flatten()
y1 = depth.flatten()
z1 = EC.copy()
fig = plt.figure()
ax1 = fig.gca()
cf1 = ax1.pcolor(x1, y1, z1*1e3, cmap=plt.cm.get_cmap('jet_r'))
cf1.set_clim(1, 100)
fig.colorbar(cf1, ax=ax1, label=r'$\sigma$ [mS/m]')
ax1.set_xlabel('X [m]')
ax1.invert_yaxis()
ax1.set_ylabel('Depth [m]')
plt.show()
