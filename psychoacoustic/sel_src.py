import numpy as np
import scipy.io.wavfile as wav
from cal_psd_thre import get_masking_thres, Transform
import torch
import argparse

def save_wav(audio, output_wav_file):
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16))

def load_wav(file_path):
    """
	returns the audio wave array, fs is assumed to be 16KHz
    """
    fs, audio = wav.read(file_path)
    if max(audio) < 1:
        audio_np = audio * 32768
    else:
        audio_np = audio
    audio_float = audio_np.astype(float)
    return audio_float, fs

class Psychoacoustic():
    def __init__(self, src_audio_path_ls, cmd_audio_path):
        self.src_audio_ls = src_audio_path_ls 
        self.n_fft = 2048
        self.win_length = 2048
        self.hop_length = 512
        cmd_audio, cmd_fs = load_wav(cmd_audio_path)
        self.cmd_audio = cmd_audio

    def sel_align_position(self, src_mask_thre, src_psd_max):
        """
	    returns the best aligned location and associated loss
        """
        transform = Transform(self.win_length)
        cmd_psd = transform(self.cmd_audio, src_psd_max)
        cmd_psd = np.transpose(cmd_psd.numpy())

        cmd_psd_shape = cmd_psd.shape[1]
        src_psd_shape = src_mask_thre.shape[1]

        align_nums = (src_psd_shape - cmd_psd_shape) + 1

        best_loss = float('inf')
        best_index = 0

        for i in range(align_nums):
            src_mask_thre_clip = src_mask_thre[:,i:i+cmd_psd_shape]
            loss = np.mean(np.maximum(cmd_psd-src_mask_thre_clip, 0))
            if loss < best_loss:
                best_loss = loss
                best_index = i
        return best_loss, best_index

    def run(self, log=None):
        """
	    returns the best aligned location and associated loss
        """
        best_loss = float('inf')
        best_time_index = 0
        best_audio_index = 0
        best_src_fs = 0
        count = 0
        for path in self.src_audio_ls:
            src_audio, src_fs = load_wav(path)

            #check if the length of source audio is shorter than the command audio
            if src_audio.shape[0] < self.cmd_audio.shape[0]:
                print("ERROR: the length of source audio is shorter than the command audio.\n")
                return "", 0, 0

            #calculate the masking threshold
            mask_thre, psd_max, time_arr = get_masking_thres(src_audio, src_fs, self.n_fft, self.win_length, self.hop_length)
            #get the best align position and loss
            loss, index = self.sel_align_position(mask_thre, psd_max)
            time_index = time_arr[index]

            if log is not None:
                log.write(path + '  ' + 'loss: ' + str(round(loss,2)) + ", aligned position: " + str(round(time_index,3)) + "s\n")

            if loss < best_loss:
                best_loss = loss
                best_time_index = time_index 
                best_audio_index = count
                best_src_fs = src_fs
            count += 1

        return self.src_audio_ls[best_audio_index], best_time_index, 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd_sound', type=str, required=True, help='commond audio path')
    parser.add_argument('--src_sound_set', type=str, required=True, help='source sound path set, seperated by comma')

    args = parser.parse_args()

    cmd_audio = args.cmd_sound

    src_sound_set = args.src_sound_set

    src_audio_list = src_sound_set.split(",")
    log_file = cmd_audio[:-4] + '_log.txt'

    psycho = Psychoacoustic(src_audio_list, cmd_audio)

    with open(log_file, 'w') as log:
        audio_name, aligned_pos, success = psycho.run(log=log)
    
    if success:
        print('Succes! The source audio name is: %s, the aligned position is: %f' % (audio_name,aligned_pos))
    else:
        print('Not totally a success! Please check the error message.')

