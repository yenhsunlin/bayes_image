import numpy as np


class Energy:
    """
    Class for calculating the energies of 256 grey values for a given pixel
    in the presence of its neighborhood
    """

    @staticmethod
    def denoise(center, nei, lam, cutoff, grey_vals = np.arange(256)):
        """
        For denosing process
        """
        Phi = (grey_vals-center)**2
        Psi = lam*(np.clip((grey_vals-nei[0,0])**2, 0, cutoff) +   \
                np.clip((grey_vals-nei[0,1])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[0,2])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[1,0])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[1,2])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[2,0])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[2,1])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[2,2])**2, 0, cutoff))
        return (Psi+Phi)
    
    @staticmethod
    def inpaint(nei, cutoff, grey_vals = np.arange(256)):
        """
        For inpainting process
        """
        Psi = (np.clip((grey_vals-nei[0,1])**2, 0, cutoff) +   \
                np.clip((grey_vals-nei[0,2])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[0,3])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[1,0])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[1,1])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[1,2])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[1,3])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[1,4])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[2,0])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[2,1])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[2,3])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[2,4])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[3,0])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[3,1])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[3,2])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[3,3])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[3,4])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[4,1])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[4,2])**2, 0, cutoff) +  \
                np.clip((grey_vals-nei[4,3])**2, 0, cutoff))
        return Psi