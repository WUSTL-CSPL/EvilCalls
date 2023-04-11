# When Evil Calls: Targeted Adversarial Voice over IP Network

## Dependencies
- torch
- numpy
- scipy
- wave
- pyaudio
- google.cloud

To install all the dependencies:
```
pip install -r requirements.txt
```
## Psychoacoustic
To determine the most proper source audio and the location to synthesize:
```
python sel_src.py --cmd_sound [cmd_audio_path] --src_sound_set [source_audio_path]
```
## Run attack
To run TAINT attack:
```
python tx.py --env_sound [env_audio_path] --tar_sound [cmd_audio_path] --tar_cmd [target command] --output_path [output audio path]  --align_pos [aligned position] --ip [ip] --port [port number]
```
```
python3 rx.py --credential_path [google_credential_file] --record_time [time_length] --ip [ip]  --port [port number] 
```
## Project website:
https://sites.google.com/view/targeted-adversarial-voip
## Citation
If you find our work useful, please cite:

```
@InProceedings{liu_evilcalls_2022,
      title = {When Evil Calls: Targeted Adversarial Voice over IP Network},
      author = {Liu, Han and Yu, Zhiyuan and Zha, Mingming and Wang, XiaoFeng and Yeoh, William and Vorobeychik, Yevgeniy and Zhang, Ning},
      booktitle = {Proceedings of the 2022 ACM SIGSAC conference on computer and communications security},
      month = {November},
      year = {2022}
}
```
