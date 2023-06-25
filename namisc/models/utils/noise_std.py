'''
Author: LOTEAT
Date: 2023-06-20 20:05:11
'''

import numpy as np

def snr2noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std