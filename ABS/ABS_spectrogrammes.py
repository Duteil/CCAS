import matplotlib.pyplot as plt
import numpy as np
import librosa
import glob
import os
import pandas as pd
import pickle

main_dir = os.path.join("D:\\", "meerkats", "Meerkat results")
model = "EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked"
run = "new_run"
spectro_path = os.path.join("C:\\", "Users", "mathi", "Desktop", "ABS presentation")

wav_path = os.path.join(main_dir, model, "trained_model", "test files")
spectro_path = os.path.join(main_path, "spectro")
wav_names = glob.glob(os.path.join(wav_path, "*" + "wav"))
wav_names.sort()

calls = [("L_VLF244_HTBL_R02_20190705-20190712_file_9_(2019_08_08-07_44_59)_85944.wav", "cc", 6601.66, 6601.71),
         ("L_VLF244_HTBL_R02_20190705-20190712_file_9_(2019_08_08-07_44_59)_85944.wav", "cc", 4326.727, 4326.802),
         ("L_VLF244_HTBL_R02_20190705-20190712_file_9_(2019_08_08-07_44_59)_85944.wav", "cc", 6499.082, 6499.145),
         ("L_VLF244_HTBL_R02_20190705-20190712_file_9_(2019_08_08-07_44_59)_85944.wav", "cc", 7514.439, 7514.512),
         ("L_VLF244_HTBL_R02_20190705-20190712_file_9_(2019_08_08-07_44_59)_85944.wav", "mo", 4544.221, 4544.368),
         ("L_VLF244_HTBL_R02_20190705-20190712_file_9_(2019_08_08-07_44_59)_85944.wav", "sn", 4546, 4546.034),
         ("HM_VHMF015_RTTB_R05_20190707-20190719_file_7_(2019_07_13-11_44_59)_135944.wav", "al", 6781.186, 6781.462)]

for i in calls:
    if i==0:
        save_name = "confused cc call.png"
    elif i<4:
        save_name = "real cc call" + str(i) + ".png"
    else:
        save_name = calls[i][1] + " example.png"
    save_path = os.path.join(spectro_path, save_name)    
    print(save_name)
    wave_data, sr = librosa.core.load(f, sr=None)
    t0 = 0
    while t0 < len(wave_data) and wave_data[t0] != 0:
        t0 += 1
    if t0 < len(wave_data):
        wave_data = wave_data[t0:min(len(wave_data),t0 + 32000)]
        fig, ax = plt.subplots()
        
        pxx, freq, t, cax = ax.specgram(wave_data, Fs=sr)
        plt.xlabel('time s')
        plt.ylabel('frequency Hz')
        cbar = fig.colorbar(cax)
        cbar.set_label('Intensity dB')
        plt.title(save_name)
        plt.xlabel('time')
        plt.specgram(wave_data, Fs=sr)
        plt.savefig(save_path, bbox_inches="tight", format="png")
        plt.close()