import sys
sys.path.append('./')
import numpy as np
from copy import copy
from energy import *


class Denoise:

    def __init__(self, im, color, lam = 1, cutoff = 1000, bit = 8):
        """
        Class for denoising noisy image. This one uses determistic algorithm.
        
        Parameter
        ---------
        im : Noisy image
        lam : lambda for penalty
        cutoff : cutoff value
        bit : color-depth, default is 8 and corresponds to 256 colors in each channel
        """
        # Preprocess the input image
        if color: # Color
            self.im = np.pad(im, ((1,1),(1,1),(0,0)), mode = 'constant')
            self.row, self.col, self.ch = self.im.shape
            self.color = color
        else: # Grey
            self.im = np.pad(im, 1, mode = 'constant')
            self.row, self.col = self.im.shape
            self.color = color
        
        self.lam = lam
        self.cutoff = cutoff
        self.bit = np.arange(2**bit)
        # Posterior image. It will be replaced by the following self.denoise_im after
        # compeleting a full denoising process. This one is used for extracting the
        # neighborhood states of every pixels. Unlike self.denoise_im, it cannot be 
        # subject to change during the denoising process until the process is done.
        self.posterior = np.array(self.im, dtype = np.int32)
        # Create an empty array for storing denoised image.
        # Its pixels will be updated dynamically during the running of denoising process. 
        self.denoise_im = np.zeros_like(self.im, dtype = np.int32)
        
    def execute(self):
        """
        Executing denoise process
        """        
        if self.color: # If image is color
            for ch in range(self.ch):
                for r in range(1,self.row-1):
                    for c in range(1,self.col-1):
                        # Calculate the energies for total 256 pixel values and pick up
                        # the value with minimum energy.
                        # This one is the correct pixel value statistically and will be
                        # saved to self.denoise_im[r,c,ch]
                        self.denoise_im[r,c,ch] = np.argmin(                                           \
                                                            Energy.denoise(self.im[r,c,ch],            \
                                                                   self.posterior[r-1:r+2,c-1:c+2,ch], \
                                                                   self.lam, self.cutoff, self.bit)    \
                                                           )              
            # Updating the posterior image by the complete self.denoise_im.
            # It will be used for calculating neighborhood state in the next iteration.
            self.posterior = self.denoise_im.copy()            
        else: # If image is grey
            for r in range(1,self.row-1):
                for c in range(1,self.col-1):
                    self.denoise_im[r,c] = np.argmin(                                        \
                                                     Energy.denoise(self.im[r,c],            \
                                                            self.posterior[r-1:r+2,c-1:c+2], \
                                                            self.lam, self.cutoff, self.bit) \
                                                    )
            self.posterior = self.denoise_im.copy()
    
    def status(self):
        """
        Return current status of denoised image
        """
        return self.denoise_im[1:self.row-1,1:self.col-1]



def bayes_denoise(im, iters = 1, lam = 1, cutoff = 2000, bit = 8, surplus = False):
    """
    Function for doing Bayesian denoising.
    
    Parameter
    ---------
    im : Noisy image
    iters : Number of iterations
    lam : lambda for penalty
    cutoff : Truncated value
    bit : Color-depth, default is 8 and corresponds to 256 colors in each channel.
          Entering depth higher than the original image will cause abnormal result
          instead of improved quality!
    surplus : Output imtermediate images of every iterations in a list
    
    Output
    ------
    A denoised image array
    """
    # Determing the input im is color or grey
    if len(im.shape) == 3: # Color
        color = True
    elif len(im.shape) == 2: # Grey
        color = False
    else: 
        raise ValueError('Input format is not an image!')
    
    # Initializing denoising process
    denoise_im = Denoise(im, color, lam, cutoff, bit)
    denoise_ls = [im.copy()]
    # Optimizing denoised image through multiple iterations
    for i in range(iters):
        denoise_im.execute()
        denoise_ls.append(denoise_im.status().copy())
    
    if not surplus:
        return denoise_ls[-1]
    else:
        return denoise_ls


def denoise_multi(im_ls):
    """
    Denoise function from multiple noisy images
    
    Parameter
    ---------
    im_ls : Image list contains multple noisy frames. Each element
            is an image array
           
    Output
    ------
    A denoised image array
    """
    im = np.float32(im_ls)
    clean = np.sum(im,axis=0)/len(im)
    return np.uint32(np.clip(clean, 0, 255))
