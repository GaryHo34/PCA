import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from scipy import io
file = "facesOlivetti.mat"
data = io.loadmat(file)

X =np.matrix(data["X"])

h = 112
w = 92

mean_face = np.mean(X, axis = 0)
X_sub = X - mean_face

k_value = 400
u, s, vt= np.linalg.svd(X_sub)

Xproj = X_sub @ vt.T

selected_face = 60
recon_err = []

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(np.reshape(X_sub[selected_face, : ], ( w, h)).T, extent=[0,1,0,1],cmap = "gray")
ax[0,0].set_axis_off()
ax[0,0].set_title("Original Image No." + str(selected_face))

for i in range(1, k_value+1):
    
    X_reconst = Xproj[selected_face, : i ] * vt[ : i ,  : ]+ mean_face
    X_err = X_reconst -  X[selected_face , : ]
    X_mse = np.mean(np.square(X_err))  
    recon_err.append(X_mse)
    
    if i == 10 :
        ax[0,1].imshow(np.reshape(X_reconst, ( w, h)).T, extent=[0,1,0,1],cmap = "gray")
        ax[0,1].set_axis_off()
        ax[0,1].set_title("Reconstructed Image with 10 bases")
        
    elif i == 100 :
        ax[1,0].imshow(np.reshape(X_reconst, ( w, h)).T, extent=[0,1,0,1],cmap = "gray")
        ax[1,0].set_axis_off()
        ax[1,0].set_title("Reconstructed Image with 100 bases")    
    
ax[1,1].plot( recon_err)
ax[1,1].set_title("Reconstruction Error")    

plt.show()
    