import sys
sys.path.append('./')
import numpy as np
from copy import copy
from energy import *

class Inpaint:
    """
    A class for restoring scratched image.
    
    Parameter
    ---------
    im : noisy image
    color : is the input image color or grey
    mask : mask image that indicates the positions of the scratched pixels
    cutoff : cutoff value
    bit : color-depth, default is 8 and corresponds to 256 colors in each channel
    """
    
    def __init__(self, im, color, mask, cutoff = 1000, bit = 8):
        """
        Define inputs and parameters
        """        
        # Preprocess the image to be inpainted
        self.color = color
        if self.color: # Color
            self.im = np.pad(im, ((2,2),(2,2),(0,0)), mode = 'constant')
            self.row, self.col, self.ch = self.im.shape
        else: # Grey
            self.im = np.pad(im, 2, mode = 'constant')
            self.row, self.col = self.im.shape
            
        # Preprocess the mask image
        self.mask = np.pad(mask, 2, mode = 'constant')
        mask_position = np.where(self.mask > 200)
        self.mask = (np.vstack((mask_position[0], mask_position[1])).T).tolist()
            
        self.cutoff = cutoff
        self.bit = np.arange(2**bit)
        
        # Prior colors for the scratched pixels
        self.prior = self.im.copy()
        if self.color: # color
            self.prior[mask_position[0],mask_position[1],] = np.random.randint(0,256,size=(len(mask_position[0]),self.ch))
        else: # grey
            self.prior[mask_position[0],mask_position[1]] = np.random.randint(0,256,size=(len(mask_position[0])))
    
    def execute(self):
        """
        Initializing the restoration process
        """         
        # Create an array for posterior
        posterior = self.prior.copy()
        
        if self.color:
            for pos in self.mask:
                r, c = pos
                for ch in range(3):
                    # Calculate the correct color from prior image and update the posterior
                    # The correcto one should have the lowest energy
                    posterior[r,c,ch] = np.argmin(Energy.inpaint(self.prior[r-2:r+3,c-2:c+3,ch],   \
                                                                 self.cutoff, self.bit))
                    #posterior[r,c,ch] = np.argmin(inpaint_energy(self.prior[r-1:r+2,c-1:c+2,ch],   \
                    #                                             self.lam, self.cutoff, self.bit))
        else:
            for pos in self.mask:
                r, c = pos
                posterior[r,c,ch] = np.argmin(Energy.inpaint(self.prior[r-2:r+3,c-2:c+3,ch],       \
                                                             self.cutoff, self.bit))
        
        # Update the prior by the posterior and can be used for the next iteration
        self.prior = posterior.copy()
                
    def status(self):
        """
        Return the current status of the restored image
        """
        return self.prior[2:self.row-2,2:self.col-2]


def bayes_inpaint(im, mask, iters = 1, cutoff = 2000, bit = 8, surplus = False):
    """
    Function for doing Bayesian denoising.
    
    Parameter
    ---------
    im : Noisy image
    mask : mask image that indicates the positions of the scratched pixels
    iters : Number of iterations
    cutoff : Truncated value
    bit : Color-depth, default is 8 and corresponds to 256 colors in each channel.
          Entering depth higher than the original image will cause abnormal result
          instead of improved quality!
    surplus : Output imtermediate images of every iterations in a list
    
    Output
    ------
    An inpainted image array
    """
    # Determing the input im is color or grey
    if len(im.shape) == 3: # Color
        color = True
    elif len(im.shape) == 2: # Grey
        color = False
    else: 
        raise ValueError('Input format is not an image!')
    
    # Is the mask a 2D arrary?
    if len(mask.shape) == 2:
        pass
    else: 
        raise ValueError('Mask format is incorrect!')
    
    # Initializing denoising process
    inpaint_im = Inpaint(im, color, mask, cutoff, bit)
    inpaint_ls = [im.copy(),inpaint_im.status().copy()]
    # Optimizing denoised image through multiple iterations
    for i in range(iters):
        inpaint_im.execute()
        inpaint_ls.append(inpaint_im.status().copy())
    
    if not surplus:
        return inpaint_ls[-1]
    else:
        return inpaint_ls