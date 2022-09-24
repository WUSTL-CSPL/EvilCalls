import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy import signal
import librosa
import scipy.io.wavfile as wav
import math
import torchaudio
import torch

def get_masking_thres(audio, fs, n_fft=2048, win_length=2048, hop_length=512):
    """
	returns the masking threshold, max PSD, and time points responding to the masking threshold
    """

    thre, psd_max = generate_th(audio, fs, n_fft, win_length, hop_length)
    thre = np.transpose(thre)
    time_arr = get_time_arr(audio, fs, hop_length, win_length)
    
    return thre, psd_max, time_arr

def compute_PSD_matrix(audio, n_fft=2048, win_length=2048, hop_length=512):
    win = np.sqrt(8.0/3.) * librosa.core.stft(audio, n_fft = n_fft, hop_length= hop_length, win_length = win_length, center=False)
    z = abs(win / win_length)
    psd_max = np.max(z*z)
    psd = 10 * np.log10(z * z + 0.0000000000000000001)
    PSD = 96 - np.max(psd) + psd
    return PSD, psd_max   

def Bark(f):
    """returns the bark-scale value for input frequency f (in Hz)"""
    return 13*np.arctan(0.00076*f) + 3.5*np.arctan(pow(f/7500.0, 2))

def quiet(f):
     """returns threshold in quiet measured in SPL at frequency f with an offset 12(in Hz)"""
     thresh = 3.64*pow(f*0.001,-0.8) - 6.5*np.exp(-0.6*pow(0.001*f-3.3,2)) + 0.001*pow(0.001*f,4) - 12
     return thresh

def two_slops(bark_psd, delta_TM, bark_maskee):
    """
	returns the masking threshold for each masker using two slopes as the spread function 
    """
    Ts = []
    for tone_mask in range(bark_psd.shape[0]):
        bark_masker = bark_psd[tone_mask, 0]
        dz = bark_maskee - bark_masker
        zero_index = np.argmax(dz > 0)
        sf = np.zeros(len(dz))
        sf[:zero_index] = 27 * dz[:zero_index]
        sf[zero_index:] = (-27 + 0.37 * max(bark_psd[tone_mask, 1] - 40, 0)) * dz[zero_index:] 
        T = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
        Ts.append(T)
    return Ts
    
def compute_th(PSD, barks, ATH, freqs):
    """ returns the global masking threshold
    """
    # Identification of tonal maskers
    # find the index of maskers that are the local maxima
    length = len(PSD)
    masker_index = signal.argrelextrema(PSD, np.greater)[0]
    
    
    # delete the boundary of maskers for smoothing
    if 0 in masker_index:
        masker_index = np.delete(0)
    if length - 1 in masker_index:
        masker_index = np.delete(length - 1)
    num_local_max = len(masker_index)

    # treat all the maskers as tonal (conservative way)
    # smooth the PSD 
    p_k = pow(10, PSD[masker_index]/10.)    
    p_k_prev = pow(10, PSD[masker_index - 1]/10.)
    p_k_post = pow(10, PSD[masker_index + 1]/10.)
    P_TM = 10 * np.log10(p_k_prev + p_k + p_k_post)
    
    # bark_psd: the first column bark, the second column: P_TM, the third column: the index of points
    _BARK = 0
    _PSD = 1
    _INDEX = 2
    bark_psd = np.zeros([num_local_max, 3])
    bark_psd[:, _BARK] = barks[masker_index]
    bark_psd[:, _PSD] = P_TM
    bark_psd[:, _INDEX] = masker_index
    
    # delete the masker that doesn't have the highest PSD within 0.5 Bark around its frequency 
    for i in range(num_local_max):
        next = i + 1
        if next >= bark_psd.shape[0]:
            break
            
        while bark_psd[next, _BARK] - bark_psd[i, _BARK]  < 0.5:
            # masker must be higher than quiet threshold
            if quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            if next == bark_psd.shape[0]:
                break
                
            if bark_psd[i, _PSD] < bark_psd[next, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            else:
                bark_psd = np.delete(bark_psd, (next), axis=0)
            if next == bark_psd.shape[0]:
                break        
    
    # compute the individual masking threshold
    delta_TM = 1 * (-6.025  -0.275 * bark_psd[:, 0])
    Ts = two_slops(bark_psd, delta_TM, barks) 
    Ts = np.array(Ts)
    
    # compute the global masking threshold
    theta_x = np.sum(pow(10, Ts/10.), axis=0) + pow(10, ATH/10.) 

    return theta_x

def generate_th(audio, fs, n_fft=2048, win_length=2048, hop_length=512):
    """
	returns the masking threshold theta_xs and the max psd of the audio
    """
    PSD, psd_max= compute_PSD_matrix(audio, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  
    freqs = librosa.core.fft_frequencies(fs, win_length)
    barks = Bark(freqs)
    # compute the quiet threshold 
    ATH = np.zeros(len(barks)) - np.inf
    bark_ind = np.argmax(barks > 1)
    ATH[bark_ind:] = quiet(freqs[bark_ind:])
    # compute the global masking threshold theta_xs 
    theta_xs = []
    # compute the global masking threshold in each window
    for i in range(PSD.shape[1]):
        theta_xs.append(compute_th(PSD[:,i], barks, ATH, freqs))
    theta_xs = np.array(theta_xs)
    return theta_xs, psd_max

def get_time_arr(audio, audio_sf, hop_length, win_length):
    """
	returns the time points in STFT array
    """
    time_nums = (len(audio) - win_length) // hop_length + 1
    time_arr = []
    for i in range(time_nums):
        time_point = (i * hop_length)/audio_sf 
        time_arr.append(time_point)
    return np.array(time_arr)

class Transform(object):
    '''
    Return: PSD
    '''    
    def __init__(self, window_size):
        self.scale = 8. / 3.
        self.frame_length = int(window_size)
        self.frame_step = int(window_size//4)
        self.window_size = window_size
    
    def __call__(self, x, psd_max_ori):

        x = torch.from_numpy(x)
        psd_max_ori = torch.tensor(psd_max_ori)

        win = torch.stft(x, self.frame_length, self.frame_step, return_complex=True, center=False)
        z = self.scale * torch.abs(win / self.window_size)
        psd = torch.square(z)
        PSD = math.pow(10., 9.6) / torch.reshape(psd_max_ori, [-1, 1, 1]) * psd
        PSD = PSD.squeeze().permute(1,0)
        return PSD