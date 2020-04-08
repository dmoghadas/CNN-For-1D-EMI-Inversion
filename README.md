# One-dimensional (1D) Deep learning (DL) inversion of loop-loop electromagnetic induction (EMI) data using convolutional neural network.

## This is the companion Python code of the paper by Moghadas GJI 2020 (see reference below). 

This code contains the following scripts: 

DLINVEMI_1D_Training: this code contains the main CNN algorithm for training EMI data. To train the network, 20,000 subsurface models were randomly generated considering 12 layers with conductivity range between 1-100 mS/m.

DLINVEMI_1D_Predictions: this code applies the trained CNN network on the EMI data (Transect 1 in the paper) measured from the Chicken Creek catchment (Brandenburg, Germany).

## Reference

Moghadas, D., 2020, One-dimensional deep learning inversion of electromagnetic induction data using convolutional neural network, Geophysical journal international, DOI: 10.1093/gji/ggaa161

## License

Apache 

## Contact

Davood Moghadas (moghadas@b-tu.de)

