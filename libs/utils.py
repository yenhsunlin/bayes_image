import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact


def psnr(im, trim):
    """
    Function for calculating PSNR index
    
    Parameter
    ---------
    im : Target image array, usually a corrupted image 
    trim : True image
    
    Output
    ------
    Scalar, PSNR index in dB unit
    """
    mse = (np.float32(im) - np.float32(trim))**2
    return 10*np.log10(255**2/np.mean(mse))


def draw_array(im, cmap=None, dpi=None):
    plt.figure(dpi=dpi)
    plt.imshow(im,cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def display_seq(im_ls, cmap=None, dpi=None):
    """
    Function for drawing sequencial figures
    """
    def _show(frame=(0,len(im_ls)-1)):
        return draw_array(im_ls[frame], cmap=cmap, dpi=dpi)
    return interact(_show)