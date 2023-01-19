# -*- coding: utf-8 -*-

#To implement TX behavior
import socket
import time
import pyaudio
import wave
import numpy as np
import scipy.io.wavfile as wav
import math
import shutil
import glob
import argparse
from tx_utils import SignOpt


def initConn(ip, port):
    sok = socket.socket()
    portRX = port
    try:
        sok.connect((ip, portRX))
    except socket.timeout:
        print("Connection timeout")
        return sok, -1
    print("Connect to RX success.")
    return sok, 0

def sync2RX(sok):
    sendStr = "Start playing."
    sok.send(bytes(sendStr, encoding = 'utf-8'))
    while True: 
        msg = sok.recv(1024)
        if msg.decode("utf-8") == "I start recording audio":
            print("Sync the TX and RX success.")
            break

def playAudio(path):
    time.sleep(0.2)
    chunk = 1024
    wf = wave.open(path, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format =
                    p.get_format_from_width(wf.getsampwidth()),
                    #channels = wf.getnchannels(),
                    channels = 1,
                    rate = wf.getframerate(),
                    output = True)

    data = wf.readframes(chunk)

    while data != b'':
        stream.write(data)
        data = wf.readframes(chunk)
        #print(data)
        
    time.sleep(1)
    stream.close()    
    p.terminate()
    #wait for the finish of the play
    wf.close()

def save_wav(audio, output_wav_file):
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16))

def load_wav(input_wav_file):
    # Load the inputs that we're given
    fs, audio = wav.read(input_wav_file)
    return audio

def synthe_wav(align_pos, env_sound, target_com, play_name, scale):
    start_index = align_pos
    result_audio = np.zeros(env_sound.shape[0])
    result_audio[:start_index] = env_sound[:start_index]
    result_audio[start_index:start_index+target_com.shape[0]] = env_sound[start_index:start_index+target_com.shape[0]] + scale * target_com
    result_audio[start_index+target_com.shape[0]:] = env_sound[start_index+target_com.shape[0]:]

    save_wav(result_audio, play_name)
    return result_audio


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_sound', type=str, required=True, help='environment sound path')
    parser.add_argument('--tar_sound', type=str, required=True, help='target sound path')
    parser.add_argument('--tar_cmd', type=str, required=True, help='target command')
    parser.add_argument('--output_path', type=str, required=True, help='output wave path')
    parser.add_argument('--align_pos', type=int, required=True, help='initial aligned position')
    parser.add_argument('--ip', type=str, required=True, help='tx ip address')
    parser.add_argument('--port', type=int, required=True, help='tx port number')
    parser.add_argument("--use_storm", type=bool, default=False, help = "if use storm optimizer")
    parser.add_argument('--play_audio_path', type=str, default="play_temp_audio.wav", help='temporary path for play audio')
    parser.add_argument('--iter_num', type=int, default=1500, help='maximum iteration number')

    args = parser.parse_args()

    print(args)

    env_sound = load_wav(args.env_sound)
    target_sound = load_wav(args.tar_sound)
    target = args.tar_cmd.upper()
    play_name = args.play_audio_path
    output_path = args.output_path
    align_pos = args.align_pos
    ip = args.ip
    port = args.port
    use_storm = args.use_storm

    scale_list = np.linspace(start=0.3, stop=0.01, num = 30)
    stage = 1
    trans = ""
    last_audio = env_sound
    env_sound = np.array(env_sound, dtype=np.float64)
    signOpt = SignOpt(env_sound, target, play_name, output_path, use_storm)
    early_flag = 0
    
    #init conn
    sok,ret = initConn(ip, port)
    if ret != 0:
        exit(-1)
    time.sleep(0.01)


    for i in range(1500):
        print("----- %d-th iteration -----" % i)
        
        #prepare audio
        if stage == 1:
            scale = scale_list[i]
            result_audio = synthe_wav(align_pos, env_sound, target_sound, play_name, scale)
        
        elif stage == 2:
            signOpt.global_binary_search(trans)
            if (signOpt.global_lbd_hi - signOpt.global_lbd_lo) < 1e-2:
                stage += 1
                #calculate inital lbd and initial theta
                signOpt.initial_lbd = signOpt.global_lbd_hi
                signOpt.xg = signOpt.initial_theta
                signOpt.gg = signOpt.global_lbd_hi
                trans = target

        elif stage == 3:
            #calculate sign-opt grad
            ret = signOpt.sign_opt_search(trans)
            if ret:
                print("Early termination due to optimization not moving.")
                early_flag = 1
                break

        sync2RX(sok)
        playAudio(play_name)

        while True:
            msg = sok.recv(1024)
            rcv_msg = msg.decode("utf-8")
            if rcv_msg[:6] == "trans:":
                print("Receive recognition decision.")
                break

        trans = rcv_msg[6:]

        if trans.upper() != target:
            if stage == 1:
                stage = 2           
                signOpt.perb_audio = last_audio
        else:
            last_audio = result_audio

    if not early_flag:
        shutil.copy("temp.wav", output_path) 
    
    print("Reach max iteration. End of the transmitter.")
    #send terminate signal to RX
    sok.send(bytes("max iteration", encoding = 'utf-8'))

 

