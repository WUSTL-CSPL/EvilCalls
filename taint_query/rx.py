# -*- coding: utf-8 -*-

#To implement RX behavior, always run first
import socket
import pyaudio
import wave
import azure.cognitiveservices.speech as speechsdk
import argparse
import string
import os

def initConn(ip, port):
    sok = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    portRX = port
    sok.bind((ip, portRX))
    sok.listen(5)
    print("waiting for any incoming connection.")
    conn, _ = sok.accept()
    print("Connection to TX success.")
    return conn
 
def sync2TX(conn):
    while True:
        msg = conn.recv(1024)
        if msg.decode("utf-8") == "Start playing.":
            conn.send(bytes("I start recording audio", encoding = 'utf-8'))
            print("Sync the TX and RX success.")
            break
        elif msg.decode("utf-8") == "max iteration":
            print("Reach max iteration. End of the receiver.")
            exit(0)

def recordAudio(path, record_time):
    #path: where to save recorded file
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100          #sampling rate
    RECORD_SECONDS = record_time   #how long the audio last, a little longer than original
    WAVE_OUTPUT_FILENAME = path

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("start recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def transSpeech(config, audio_path):
    with open(audio_path, 'rb') as f:
        byte_wave_f = f.read()
    audio_wav = speech.RecognitionAudio(content=byte_wave_f)
    client = speech.SpeechClient()
    response = client.recognize(config=config, audio=audio_wav)
    for result in response.results:
        best_alternative = result.alternatives[0]
        transcript = best_alternative.transcript
        return transcript


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=True, help='tx ip address')
    parser.add_argument('--port', type=int, required=True, help='tx port number')
    parser.add_argument('--record_time', type=float, required=True, help='recorded time length in seconds')
    parser.add_argument('--credential_path', type=float, required=True, help='Google credential file path')
    parser.add_argument('--record_audio_path', type=str, default='record.wav', help='temporary path for record audio')

    args = parser.parse_args()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
    ip = args.ip
    port = args.port
    rec_file_path = args.record_audio_path
    record_time = args.record_time
    config = dict(language_code="en-US", max_alternatives=10, sample_rate_hertz=16000, model = "command_and_search")
    conn = initConn(ip, port)

    while True:
        sync2TX(conn)
        recordAudio(rec_file_path, record_time)
        trans = transSpeech(config, rec_file_path)

        print("transription: ", trans)

        decision = "trans:" + str(trans)
        #send decisions
        conn.send(bytes(decision, encoding = 'utf-8'))


