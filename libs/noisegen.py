import numpy as np
from copy import copy


def gaussian_noise(im, mean = 0, sd = 20):
    """
    Function for generating Gaussian noise on data.
    
    Parameter
    ---------
    im : array-like
    mean : noise mean
    sd : Noise standard deviation
    
    Output
    ------
    Array-like, original data plus Gaussian noise
    """
    im = np.asarray(im)
    size = im.shape
    #noise = np.random.normal(loc = mean, scale = sd, size = (size[0],size[1],1))
    noise = np.random.normal(loc = mean, scale = sd, size = size)
    #if len(size) == 3:
    #    noise_im = np.clip(im + noise/3, 0, 255) # chop values outside [0,255]
    #else:
    #    noise_im =np.clip(im + noise[:,:,0], 0, 255)    
    noise_im = np.clip(im + noise, 0, 255) # chop values outside [0,255]
    return np.uint8(noise_im)


def uniform_noise(im, amp = 10):
    """
    Function for generating uniform noise with zero mean
    on data
    
    Parameter
    ---------
    im : array-like
    amp : positive scalar, noise amplitude
    
    Output
    ------
    Array-like, original data plus uniform noise
    """
    im = np.asarray(im)
    noise = np.random.uniform(low = -amp, high = amp, size = im.shape[:2])
    if len(noise.shape) == 3:
        noise_im = np.clip(im + noise/3, 0, 255) # chop values outside [0,255]
    else:
        noise_im =np.clip(im + noise, 0, 255)
    return np.uint8(noise_im)


def poisson_noise(im, a = 20):
    """
    Function for generating Poisson noise on data
    
    Parameter
    ---------
    im : array-like
    a : amount of Poisson noise
    
    Output
    ------
    Array-like, original data plus Poisson noise
    """
    # Poisson noise can be modeled by Gaussian dist. Poi~Gauss(num*t,num*t)
    # where t is the unit time. Here we set t = 1
    im = np.asarray(im)
    size = im.shape
    #noise = np.random.normal(loc = a, scale = a, size = im.shape[:2])
    #if len(noise.shape) == 3:
    #    noise_im = np.clip(im + noise/3, 0, 255) # chop values outside [0,255]
    #else:
    #    noise_im =np.clip(im + noise, 0, 255)
    noise = np.random.normal(loc = a, scale = a, size = size)
    noise_im = np.clip(im + noise, 0, 255)
    return np.uint8(noise_im)


def saltpepper_noise(im, noi_f = 0.2, pep_f = 0.5):
    """
    Function for generating salt-and-pepper noise on data
    
    Parameter
    ---------
    im : array-like
    noi_f : Fraction of the image having salt-and-peppr noise
    pep_f : Fraction of the noise being pepper, while the
            remaining 1-pep_f will be salt
    
    Output
    ------
    Array-like, original data plus salt-and-pepper noise
    """
    img = (np.asarray(im)).copy()
    #img = im.copy()
    if (0 <= noi_f <= 1) and (0 <= pep_f <= 1):
        # Tokens 0 for pepper, 255 for salt, 1 for doing nothing
        token = np.random.choice([0,255,1],p=[noi_f*pep_f,noi_f*(1-pep_f),1-noi_f],size=img.shape[:2])
        img[token == 0] = 0
        img[token == 255] = 255
        return np.uint8(img)
    else:
        raise ValueError('Noise and salt fractions must lie within 0 and 1.')
