# %%
from tqdm import tqdm
import os
import numpy as np

np.set_printoptions(threshold=np.inf)

from scipy.special._ufuncs import gamma


# %%

# # WAVELET  1D Wavelet transform with optional significance testing
#   wave, period, scale, coi = wavelet(Y, dt, pad, dj, s0, J1, mother, param)
#
#   Computes the wavelet transform of the vector Y (length N),
#   with sampling rate DT.
#
#   By default, the Morlet wavelet (k0=6) is used.
#   The wavelet basis is normalized to have total energy=1 at all scales.
#
# INPUTS:
#
#    Y = the time series of length N.
#    DT = amount of time between each Y value, i.e. the sampling time.
#
# OUTPUTS:
#
#    WAVE is the WAVELET transform of Y. This is a complex array
#    of dimensions (N,J1+1). FLOAT(WAVE) gives the WAVELET amplitude,
#    ATAN(IMAGINARY(WAVE),FLOAT(WAVE) gives the WAVELET phase.
#    The WAVELET power spectrum is ABS(WAVE)**2.
#    Its units are sigma**2 (the time series variance).
#
# OPTIONAL INPUTS:
#
# *** Note *** if none of the optional variables is set up, then the program
#   uses default values of -1.
#
#    PAD = if set to 1 (default is 0), pad time series with zeroes to get
#         N up to the next higher power of 2. This prevents wraparound
#         from the end of the time series to the beginning, and also
#         speeds up the FFT's used to do the wavelet transform.
#         This will not eliminate all edge effects (see COI below).
#
#    DJ = the spacing between discrete scales. Default is 0.25.
#         A smaller # will give better scale resolution, but be slower to plot.
#
#    S0 = the smallest scale of the wavelet.  Default is 2*DT.
#
#    J1 = the # of scales minus one. Scales range from S0 up to S0*2**(J1*DJ),
#        to give a total of (J1+1) scales. Default is J1 = (LOG2(N DT/S0))/DJ.
#
#    MOTHER = the mother wavelet function.
#             The choices are 'MORLET', 'PAUL', or 'DOG'
#
#    PARAM = the mother wavelet parameter.
#            For 'MORLET' this is k0 (wavenumber), default is 6.
#            For 'PAUL' this is m (order), default is 4.
#            For 'DOG' this is m (m-th derivative), default is 2.
#
#
# OPTIONAL OUTPUTS:
#
#    PERIOD = the vector of "Fourier" periods (in time units) that corresponds
#           to the SCALEs.
#
#    SCALE = the vector of scale indices, given by S0*2**(j*DJ), j=0...J1
#            where J1+1 is the total # of scales.
#
#    COI = if specified, then return the Cone-of-Influence, which is a vector
#        of N points that contains the maximum period of useful information
#        at that particular time.
#        Periods greater than this are subject to edge effects.
def cwt(Y, dt, pad=0, dj=-1, s0=-1, J1=-1, mother=-1, param=6, freq=None):
    n = len(Y)

    if s0 == -1:
        s0 = 2 * dt
    if dj == -1:
        dj = 1. / 4.
    if J1 == -1:
        J1 = np.fix((np.log(n * dt / s0) / np.log(2)) / dj)
    if mother == -1:
        mother = 'MORLET'

    # construct time series to analyze, pad if necessary
    x = Y - np.mean(Y)
    if pad == 1:
        # power of 2 nearest to N
        base2 = np.fix(np.log(n) / np.log(2) + 0.4999)
        nzeroes = (2 ** (base2 + 1) - n).astype(np.int64)
        x = np.concatenate((x, np.zeros(nzeroes)))

    # construct wavenumber array used in transform [Eqn(5)]
    kplus = np.arange(1, int(n / 2) + 1)
    kplus = (kplus * 2 * np.pi / (n * dt))
    kminus = np.arange(1, int((n - 1) / 2) + 1)
    kminus = np.sort((-kminus * 2 * np.pi / (n * dt)))
    k = np.concatenate(([0.], kplus, kminus))

    # compute FFT of the (padded) time series
    f = np.fft.fft(x)  # [Eqn(3)]

    # construct SCALE array & empty PERIOD & WAVE arrays
    if mother.upper() == 'MORLET':
        if param == -1:
            param = 6.
        fourier_factor = 4 * np.pi / (param + np.sqrt(2 + param ** 2))
    elif mother.upper() == 'PAUL':
        if param == -1:
            param = 4.
        fourier_factor = 4 * np.pi / (2 * param + 1)
    elif mother.upper() == 'DOG':
        if param == -1:
            param = 2.
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * param + 1))
    else:
        fourier_factor = np.nan

    if freq is None:
        j = np.arange(0, J1)
        scale = s0 * 2. ** (j * dj)
        freq = 1. / (fourier_factor * scale)
        period = 1. / freq
    else:
        scale = 1. / (fourier_factor * freq)
        period = 1. / freq
    # define the wavelet array
    wave = np.zeros(shape=(len(scale), n), dtype=complex)

    # loop through all scales and compute transform
    for a1 in range(0, len(scale)):
        daughter, fourier_factor, coi, _ = \
            wave_bases(mother, k, scale[a1], param)
        wave[a1, :] = np.fft.ifft(f * daughter)  # wavelet transform[Eqn(4)]

    # COI [Sec.3g]
    coi = coi * dt * np.concatenate((
        np.insert(np.arange(int((n + 1) / 2) - 1), [0], [1E-5]),
        np.insert(np.flipud(np.arange(0, int(n / 2) - 1)), [-1], [1E-5])))
    wave = wave[:, :n]  # get rid of padding before returning

    # return wave, period, scale, coi
    return wave


# --------------------------------------------------------------------------
# WAVE_BASES  1D Wavelet functions Morlet, Paul, or DOG
#
#  DAUGHTER,FOURIER_FACTOR,COI,DOFMIN = wave_bases(MOTHER,K,SCALE,PARAM)
#
#   Computes the wavelet function as a function of Fourier frequency,
#   used for the wavelet transform in Fourier space.
#   (This program is called automatically by WAVELET)
#
# INPUTS:
#
#    MOTHER = a string, equal to 'MORLET' or 'PAUL' or 'DOG'
#    K = a vector, the Fourier frequencies at which to calculate the wavelet
#    SCALE = a number, the wavelet scale
#    PARAM = the nondimensional parameter for the wavelet function
#
# OUTPUTS:
#
#    DAUGHTER = a vector, the wavelet function
#    FOURIER_FACTOR = the ratio of Fourier period to scale
#    COI = a number, the cone-of-influence size at the scale
#    DOFMIN = a number, degrees of freedom for each point in the wavelet power
#             (either 2 for Morlet and Paul, or 1 for the DOG)

def wave_bases(mother, k, scale, param):
    n = len(k)
    kplus = np.array(k > 0., dtype=float)

    if mother == 'MORLET':  # -----------------------------------  Morlet

        if param == -1:
            param = 6.

        k0 = np.copy(param)
        # calc psi_0(s omega) from Table 1
        expnt = -(scale * k - k0) ** 2 / 2. * kplus
        norm = np.sqrt(scale * k[1]) * (np.pi ** (-0.25)) * np.sqrt(n)
        daughter = norm * np.exp(expnt)
        daughter = daughter * kplus  # Heaviside step function
        # Scale-->Fourier [Sec.3h]
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
        coi = fourier_factor / np.sqrt(2)  # Cone-of-influence [Sec.3g]
        dofmin = 2  # Degrees of freedom
    elif mother == 'PAUL':  # --------------------------------  Paul
        if param == -1:
            param = 4.
        m = param
        # calc psi_0(s omega) from Table 1
        expnt = -scale * k * kplus
        norm_bottom = np.sqrt(m * np.prod(np.arange(1, (2 * m))))
        norm = np.sqrt(scale * k[1]) * (2 ** m / norm_bottom) * np.sqrt(n)
        daughter = norm * ((scale * k) ** m) * np.exp(expnt) * kplus
        fourier_factor = 4 * np.pi / (2 * m + 1)
        coi = fourier_factor * np.sqrt(2)
        dofmin = 2
    elif mother == 'DOG':  # --------------------------------  DOG
        if param == -1:
            param = 2.
        m = param
        # calc psi_0(s omega) from Table 1
        expnt = -(scale * k) ** 2 / 2.0
        norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
        daughter = -norm * (1j ** m) * ((scale * k) ** m) * np.exp(expnt)
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 1
    else:
        print('Mother must be one of MORLET, PAUL, DOG')

    return daughter, fourier_factor, coi, dofmin


# %%
def compress(data, mult=50):
    shape = list(data.shape)
    shape[-1] = mult
    shape.append(-1)
    data = np.reshape(data, shape)
    exp = np.mean(data, axis=-1)
    exp = (exp - np.min(exp)) / (np.max(exp) - np.min(exp))
    var = np.var(data, axis=-1)
    var = (var - np.min(var)) / (np.max(var) - np.min(var))
    return exp, var


# %%
if __name__ == '__main__':
    source_path = r'D:\BJM\SHHS-1'
    target_path = r'D:\BJM\SHHS-clean'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    dirs = [os.listdir(source_path)[0]]
    # dirs.reverse()
    pbar = tqdm(dirs, ncols=100)
    for i, dir in enumerate(pbar):
        with np.load(os.path.join(source_path, dir)) as f:
            data = f['x']
            label = f['y']
        spectrum = []
        for d in data:
            wave = cwt(np.squeeze(d), dt=1 / 100, J1=30, dj=0.25)
            s = np.log2(np.abs(wave) ** 2)
            s = compress(s, 120)[0]
            s = np.expand_dims(s, 0)
            spectrum.append(s)
        spectrum = np.asarray(spectrum, dtype=np.float16)
        save_dict = {
            # 'x':np.expand_dims(f['x'], 1),
            'x': spectrum,
            'y': label,
        }
        np.savez(os.path.join(target_path, dir), **save_dict)
        pbar.set_postfix({'shape': spectrum.shape})
