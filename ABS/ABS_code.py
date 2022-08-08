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

def list_files(directory, ext=".txt"):
    "list_files(directory) - Grab all .txt or specified extension files in specified directory"
    files = glob.glob(os.path.join(directory, "*" + ext))
    files.sort()
    return files

def draw_spectro(audio_file, call, start, end, save_path, title):
    '''
    Draw a spectrogram of a given section of an audio file. 
    '''
    wave_data, sr = librosa.core.load(os.path.join(audio_file), sr=None)
    wave_data = wave_data[start:min(len(wave_data),round(start*sr) + 32000)]
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

if __name__=="__main__":
    from predation_params import run_name, results_dir, audio_dirs, data, low_thresholds, high_thresholds
    
    wav_dir = os.path.join("D:\\", "meerkats", "Meerkat results", "EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked", "trained_model", "test files")

    wav_name = ["HM_VCVM001_HMB_R11_20170821-20170825_file_6_(2017_08_25-06_44_59)_ASWMUX221163",
                "HM_VHMF001_HTB_R07_20170802-20170815_file_5_(2017_08_06-06_44_59)_ASWMUX221092",
                "HM_VHMF001_HTB_R20_20190707-20190719_file_10_(2019_07_16-11_44_59)_165944",
                "HM_VHMF001_HTB_R20_20190707-20190719_file_7_(2019_07_13-11_44_59)_135944",
                "HM_VHMF015_RTTB_R05_20190707-20190719_file_10_(2019_07_16-11_44_59)_165944",
                "HM_VHMF015_RTTB_R05_20190707-20190719_file_7_(2019_07_13-11_44_59)_135944",
                "HM_VHMF019_MBTB_R25_20190707-20190719_file_10_(2019_07_16-11_44_59)_165944",
                "HM_VHMF019_MBTB_R25_20190707-20190719_file_8_(2019_07_14-11_44_59)_145944",
                "HM_VHMF022_MBRS_R22_20190707-20190719_file_13_(2019_07_19-11_44_59)_195944",
                "HM_VHMF022_MBRS_R22_20190707-20190719_file_6_(2019_07_12-11_44_59)_125944",
                "HM_VHMF022_MBRS_R22_20190707-20190719_file_8_(2019_07_14-11_44_59)_145944",
                "HM_VHMM002_HRT_R09_20170821-20170825_file_5_(2017_08_24-06_44_59)_ASWMUX221110",
                "HM_VHMM006_RT_R10_20170903-20170908_file_2_(2017_09_03-05_44_59)_ASWMUX221102",
                "HM_VHMM006_RT_R12_20170821-20170825_file_6_(2017_08_25-06_44_59)_ASWMUX221102",
                "HM_VHMM007_LSLT_R17_20190707-20190719_file_12_(2019_07_18-11_44_59)_185944",
                "HM_VHMM008_SHTB_R14_20190707-20190719_file_7_(2019_07_13-11_44_59)_135944",
                "HM_VHMM016_LTTB_R29_20190707-20190719_file_7_(2019_07_13-11_44_59)_135944",
                "HM_VHMM017_RSTB_R23_20190708-20190720_file_11_(2019_07_17-11_44_59)_175944",
                "HM_VHMM021_MBLT_R01_20190707-20190719_file_12_(2019_07_18-11_44_59)_185944",
                "HM_VHMM021_MBLT_R01_20190707-20190719_file_13_(2019_07_19-11_44_59)_195944",
                "HM_VHMM021_MBLT_R01_20190707-20190719_file_6_(2019_07_12-11_44_59)_125944",
                "HM_VHMM021_MBLT_R01_20190707-20190719_file_8_(2019_07_14-11_44_59)_145944",
                "L_VLF235_RSRTTBL_R26_20190805-20190812_file_8_(2019_08_07-07_44_59)_75944",
                "L_VLF235_RSRTTBL_R26_20190805-20190812_file_9_(2019_08_08-07_44_59)_85944",
                "L_VLF244_HTBL_R02_20190805-20190812_file_9_(2019_08_08-07_44_59)_85944",
                "L_VLF246_LTTBL_R14_20190805-20190812_file_8_(2019_08_07-07_44_59)_75944",
                "L_VLF246_LTTBL_R14_20190805-20190812_file_9_(2019_08_08-07_44_59)_85944",
                "L_VLM234_SHMBTBL_R11_20190805-20190812_file_10_(2019_08_09-07_44_59)_95944",
                "L_VLM234_SHMBTBL_R11_20190805-20190812_file_7_(2019_08_06-07_44_59)_65944",
                "L_VLM239_HTB_R10_20190805-20190812_file_8_(2019_08_07-07_44_59)_75944"] 
    wav_name.sort()
    
    radical = []
    for file in wav_name:
        radical.append(file[:-4])
        
    indices_list = [2, 3, 4, 6, 8, 10, 14, 18, 19]
    
    main_dir =  os.path.join(results_dir, run_name, data)
    label_dir = os.path.join(main_dir, "label_table")
    label_list = list_files(label_dir)
    main_metric = os.path.join("D:\\", "meerkats", "Meerkat results", "EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked", "new_run", "metrics")
    call_types = {'cc', 'sn', 'mo', 'agg', 'ld', 'soc',  'al', 'beep', 'synch', 'oth', 'noise'}
    short_GT_removed = [0.0, 0.005, 0.015, 0.02, 0.025, 0.03, "none"] 
                  
    '''We want to plot precision/recall graphs for strict and lenient,
    to plot the spectrogram of the most confused call,
    to extract the time of the worst offset (negative, positive and from the end).
    '''
                    
    highest_neg_offset = 0
    highest_pos_offset = 0
    highest_neg_onset = 0
    highest_pos_onset = 0
    largest_confusion = 0
    most_fragments = 0
    
    for GT_proportion_cut in short_GT_removed: #,0.005,0.01,0.015,0.02,0.025,0.03]:
        for low_thresh in low_thresholds:
            for high_thresh in high_thresholds:
                if low_thresh < high_thresh:
                    print([GT_proportion_cut, low_thresh, high_thresh])

                    thr_path = os.path.join(main_metric, str(GT_proportion_cut), str(low_thresh), str(high_thresh))
                    
                    cat_frag = pickle.load(open(os.path.join(thr_path, "Category_fragmentation.p"), "rb"))
                    offset = pickle.load(open(os.path.join(thr_path, "Time_Difference_start.p"), "rb"))
                    onset = pickle.load(open(os.path.join(thr_path, "Time_Difference_start.p"), "rb"))
                    match = pickle.load(open(os.path.join(thr_path, "Matching_Table.p"), "rb"))
                    call_match = pickle.load(open(os.path.join(thr_path, "Matching_Table-Labels_Sorted.p"), "rb"))
                    gt_indices = pickle.load(open(os.path.join(thr_path, "Label_Indices.p"), "rb"))
                    pred_indices = pickle.load(open(os.path.join(thr_path, "Prediction_Indices.p"), "rb"))
                    time_frag = pickle.load(open(os.path.join(thr_path, "Time_fragmentation.p"), "rb"))
                    index_list = gt_indices.index
                    
                    # computing the offset:
                    offset = [None] *len(match)
                    for idx in range(len(gt_indices)):
                        offset[idx] = pd.DataFrame(columns = call_types, index = call_types)
                        for call in call_types.difference({'noise'}):#,'synch','oth'}):
                            for pred in call_types.difference({'noise'}):#,'synch','oth'}):
                                offset[idx].at[call,pred] = []
                                for pair_nb in range(len(match[idx].at[call,pred])):
                                    call_start = gt_indices[call][index_list[idx]][match[idx].at[call,pred][pair_nb][0]][0]
                                    detected_start = pred_indices[pred][index_list[idx]][match[idx].at[call,pred][pair_nb][1]][0]
                                    offset[idx].at[call,pred].append(call_start - detected_start)
                    
                    # computing the onset:
                    onset = [None] *len(match)
                    for idx in range(len(gt_indices)):
                        onset[idx] = pd.DataFrame(columns = call_types, index = call_types)
                        for call in call_types.difference({'noise'}):#,'synch','oth'}):
                            for pred in call_types.difference({'noise'}):#,'synch','oth'}):
                                onset[idx].at[call,pred] = []
                                for pair_nb in range(len(match[idx].at[call,pred])):
                                    call_end = gt_indices[call][index_list[idx]][match[idx].at[call,pred][pair_nb][0]][1]
                                    detected_end = pred_indices[pred][index_list[idx]][match[idx].at[call,pred][pair_nb][1]][1]
                                    onset[idx].at[call,pred].append(call_end - detected_end)
                                    
                    with open(os.path.join(thr_path, 'Time_Difference_end.p'), 'wb') as fp:
                        pickle.dump(onset, fp)    
                    with open(os.path.join(thr_path, 'Time_Difference_start.p'), 'wb') as fp:
                        pickle.dump(offset, fp)      



                    files = index_list
                    metrics_folder = thr_path
                    
                    #offset
                    offset_dict = dict()
                    offset_ref = dict()
                    neg_offset_dict = dict()
                    neg_offset_ref = dict()        
                    for call in call_types.difference({'noise','synch','oth'}):
                        offset_dict[call] = 0
                        neg_offset_dict[call] = 0
                        offset_ref[call] = np.nan
                        neg_offset_ref[call] = np.nan
                    for call in call_types.difference({'noise','synch','oth'}):
                        for idx in range(len(offset)):
                            for pred in call_types.difference({'noise','synch','oth'}):
                                for match_num in range(len(offset[idx].at[call,pred])):
                                    if offset[idx].at[call,pred][match_num] < 0:
                                        if offset[idx].at[call,pred][match_num] < neg_offset_dict[call]:
                                            neg_offset_dict[call] = offset[idx].at[call,pred][match_num]
                                            neg_offset_ref[call] = (files[idx], pred, gt_indices.at[files[idx], call][match[idx].at[call,pred][match_num][0]][0], pred_indices.at[files[idx],pred][match[idx].at[call,pred][match_num][1]][0], - offset[idx].at[call,pred][match_num])
                                    else:
                                        if offset[idx].at[call,pred][match_num] > offset_dict[call]:
                                            offset_dict[call] = offset[idx].at[call,pred][match_num]
                                            offset_ref[call] = (files[idx], pred, gt_indices.at[files[idx], call][match[idx].at[call,pred][match_num][0]][0], pred_indices.at[files[idx],pred][match[idx].at[call,pred][match_num][1]][0], offset[idx].at[call,pred][match_num])
                                        
                        if offset_dict[call] > highest_pos_offset:
                            offset1 = offset_ref[call]
                            highest_pos_offset = offset_dict[call]
                        if neg_offset_dict[call] > highest_pos_offset:
                            offset2 = neg_offset_ref[call]
                            highest_neg_offset = offset_dict[call]                            

                    #onset
                    onset_dict = dict()
                    onset_ref = dict()
                    neg_onset_dict = dict()
                    neg_onset_ref = dict()        
                    for call in call_types.difference({'noise','synch','oth'}):
                        onset_dict[call] = 0
                        neg_onset_dict[call] = 0
                        onset_ref[call] = np.nan
                        neg_onset_ref[call] = np.nan
                    for call in call_types.difference({'noise','synch','oth'}):
                        for idx in range(len(onset)):
                            for pred in call_types.difference({'noise','synch','oth'}):
                                for match_num in range(len(onset[idx].at[call,pred])):
                                    if onset[idx].at[call,pred][match_num] < 0:
                                        if onset[idx].at[call,pred][match_num] < neg_onset_dict[call]:
                                            neg_onset_dict[call] = onset[idx].at[call,pred][match_num]
                                            neg_onset_ref[call] = (files[idx], pred, gt_indices.at[files[idx], call][match[idx].at[call,pred][match_num][0]][1], pred_indices.at[files[idx],pred][match[idx].at[call,pred][match_num][1]][1], - onset[idx].at[call,pred][match_num])
                                    else:
                                        if onset[idx].at[call,pred][match_num] > onset_dict[call]:
                                            onset_dict[call] = onset[idx].at[call,pred][match_num]
                                            onset_ref[call] = (files[idx], pred, gt_indices.at[files[idx], call][match[idx].at[call,pred][match_num][0]][1], pred_indices.at[files[idx],pred][match[idx].at[call,pred][match_num][1]][1], onset[idx].at[call,pred][match_num])      
                        
                        if onset_dict[call] > highest_pos_onset:
                            onset1 = onset_ref[call]
                            highest_pos_onset = onset_dict[call]
                        if neg_onset_dict[call] > highest_pos_onset:
                            onset2 = onset_ref[call]
                            highest_neg_onset = onset_dict[call]                       
