################################################################################
# BSD 3-Clause License

# Copyright (c) 2019, NVIDIA Corporation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################
# Reference: https://github.com/NVIDIA/mellotron/blob/master/yin.py

# adapted from https://github.com/patriceguyot/Yin


import numpy as np


def differenceFunction(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]
    This solution is implemented directly with Numpy fft.


    :param x: audio data
    :param N: length of data
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """

    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv


def cumulativeMeanNormalizedDifferenceFunction(df, N):
    """
    Compute cumulative mean normalized difference function (CMND).

    This corresponds to equation (8) in [1]

    :param df: Difference function
    :param N: length of data
    :return: cumulative mean normalized difference function
    :rtype: list
    """

    cmndf = df[1:] * range(1, N) / (np.cumsum(df[1:]).astype(float) + 1e-8) #scipy method
    return np.insert(cmndf, 0, 1)


def getPitch(cmdf, tau_min, tau_max, harmo_th=0.1):
    """
    Return fundamental period of a frame based on CMND function.

    :param cmdf: Cumulative Mean Normalized Difference function
    :param tau_min: minimum period for speech
    :param tau_max: maximum period for speech
    :param harmo_th: harmonicity threshold to determine if it is necessary to compute pitch frequency
    :return: fundamental period if there is values under threshold, 0 otherwise
    :rtype: float
    """
    tau = tau_min
    while tau < tau_max:
        if cmdf[tau] < harmo_th:
            while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                tau += 1
            return tau
        tau += 1

    return 0    # if unvoiced

# Heejo added pad parameters to match the length to the librosa stft func.
def compute_yin(sig, sr, w_len=512, w_step=256, f0_min=100, f0_max=500,
                harmo_thresh=0.1, center = True, pad_mode='reflect', n_fft=2048):
    """

    Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.

    :param sig: Audio signal (list of float)
    :param sr: sampling rate (int)
    :param w_len: size of the analysis window (samples)
    :param w_step: size of the lag between two consecutives windows (samples)
    :param f0_min: Minimum fundamental frequency that can be detected (hertz)
    :param f0_max: Maximum fundamental frequency that can be detected (hertz)
    :param harmo_tresh: Threshold of detection. The yalgorithmù return the first minimum of the CMND function below this treshold.

    :returns:

        * pitches: list of fundamental frequencies,
        * harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
        * argmins: minimums of the Cumulative Mean Normalized DifferenceFunction
        * times: list of time of each estimation
    :rtype: tuple
    """
    if center:
        sig = np.pad(sig, (w_step + w_len - sig.shape[0] % w_step) // 2, mode=pad_mode)

    tau_min = int(sr / f0_max)
    tau_max = int(sr / f0_min)

    timeScale = range(0, len(sig) - w_len, w_step)  # time values for each analysis window
    times = [t/float(sr) for t in timeScale]
    frames = [sig[t:t + w_len] for t in timeScale]

    pitches = [0.0] * len(timeScale)
    harmonic_rates = [0.0] * len(timeScale)
    argmins = [0.0] * len(timeScale)

    for i, frame in enumerate(frames):        
        # Compute YIN
        df = differenceFunction(frame, w_len, tau_max)
        cmdf = cumulativeMeanNormalizedDifferenceFunction(df, tau_max)
        p = getPitch(cmdf, tau_min, tau_max, harmo_thresh)

        # Get results
        if np.argmin(cmdf) > tau_min:
            argmins[i] = float(sr / np.argmin(cmdf))
        if p != 0:  # A pitch was found
            pitches[i] = float(sr / p)
            harmonic_rates[i] = cmdf[p]
        else:  # No pitch, but we compute a value of the harmonic rate
            harmonic_rates[i] = min(cmdf)

    return np.array(pitches), np.array(harmonic_rates), argmins, times





# This part is added from CODEJIN, Jaekoo
from scipy.ndimage import gaussian_filter1d
def pitch_calc(
    sig,
    sr,
    w_len=1024,
    w_step=256,
    f0_min=100,
    f0_max=500,
    confidence_threshold=0.85,
    gaussian_smoothing_sigma = 1.0
    ):
    pitch = compute_yin(
        sig= sig,
        sr= sr,
        w_len= 1024,
        w_step= 256,
        harmo_thresh= 1 - confidence_threshold
        )[0]
    if gaussian_smoothing_sigma > 0.0:
        pitch = gaussian_filter1d(pitch, sigma= gaussian_smoothing_sigma)
    
    return pitch
    