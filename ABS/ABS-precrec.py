import matplotlib.pyplot as plt
import librosa
import numpy as np
import glob
import os
import tensorflow as tf
import pandas as pd
import pickle
import ntpath

from predation_params import *

main_dir = os.path.join("D:\\", "meerkats", "Meerkat results")
model = "EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked"
run = "new_run"
metrics_folder = os.path.join(main_dir, model, run, "metrics") #, "call_type_by_call_type", "default mode")
save_folder = os.path.join("C:\\", "Users", "mathi", "Desktop", "ABS presentation")

true_calls = {'agg', 'al', 'cc', 'ld', 'mo', 'sn', 'soc'}
call_types = {'agg', 'al', 'cc', 'ld', 'mo', 'sn', 'soc', 'synch', 'beep', 'oth'}
# all_thr = [0.3, 0.4,0.5,0.6,0.7,0.8,0.9,0.95]
row0 = pd.DataFrame(columns = true_calls) #, index=range(len(tablenames)))
row1 = pd.DataFrame(columns = true_calls)
for call in true_calls:
    row0.at[0,call] = 0
    row1.at[0,call] = 1
row1.rename(index={0: 1}, inplace=True)

for GT_proportion_cut in [0.0]: #short_GT_removed: #,0.005,0.01,0.015,0.02,0.025,0.03]:
    for low_thresh in [0.3]:#low_thresholds:
        if low_thresh >= high_thresholds[0]:
            del high_thresholds[0]
        print([GT_proportion_cut, low_thresh])
        # for low_thr in [0.2, 0.3]:
        #     while all_thr[0] <= low_thr:
        #         del all_thr[0]  
        TP = pd.DataFrame(columns = true_calls, index=high_thresholds)
        FP = pd.DataFrame(columns = true_calls, index=high_thresholds)
        FN = pd.DataFrame(columns = true_calls, index=high_thresholds)
        prec = pd.DataFrame(columns = true_calls, index=high_thresholds)
        rec = pd.DataFrame(columns = true_calls, index=high_thresholds)
        strict_prec = pd.DataFrame(columns = true_calls, index=high_thresholds)
        strict_rec = pd.DataFrame(columns = true_calls, index=high_thresholds)
