#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from python_speech_features import mfcc
from keras.models import load_model

from configparser import ConfigParser


# In[6]:

file_path = sys.argv[1]

config = ConfigParser()
config.read('config.ini')

sound = config.get('main', "sound")
nfilt = config.getint('main', "nfilt")
nfeat = config.getint('main', 'nfeat')
nfft = config.getint('main', 'nfft')
rate = config.getint('main', 'rate')
step = config.getint('main', 'step')

train_folder = config.get('main', 'train_folder')
test_folder = config.get('main', 'test_folder')
train_descrption = config.get('main', 'train_description')
test_description = config.get('main', 'test_description')
global_min = config.getfloat('main', 'global_min') 
global_max = config.getfloat('main', 'global_max')
model_path = config.get('main', 'model_path')


# In[19]:


model = load_model(model_path)


# In[7]:


def single_prediction(filepath):
    
    fn_prob = []
    match_plot = []
    timestamps = []
    merged_timestamps = []
    y_prob = []

    rate, wav = wavfile.read(filepath)
    
    for i in range(0, wav.shape[0]-step, step):
        sample = wav[i:i+step]
        x = mfcc(sample, rate, numcep=nfeat, 
                 nfilt=nfilt, nfft=nfft).T

        x = (x - global_min) / (global_max - global_min)

        x = x.reshape(1, x.shape[0], x.shape[1], 1)
        y_hat = model.predict(x)
        
        if(y_hat[0][1] > y_hat[0][0]):
            match_plot.append(1)
            timestamps.append((i/rate , (i + step)/rate))
        else:
            match_plot.append(0)
            
        y_prob.append(y_hat)

    for i in range(len(timestamps)):
        if(i > 0 and timestamps[i-1][1] == timestamps[i][0]):
            tmp = merged_timestamps.pop()
            merged_timestamps.append((tmp[0], timestamps[i][1]))
        else:
            merged_timestamps.append((timestamps[i][0], timestamps[i][1]))
    
    fn_prob.append(np.mean(y_prob, axis=0).flatten())
    return fn_prob, match_plot, wav, merged_timestamps


# In[8]:


def find_contiguous_colors(colors):
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg)
    return segs
 
def plot_multicolored_lines(x,y,colors, fn_prob, stamps):

    message = ""

    for item in stamps:
        message += "Found match at " + str(item[0]) + " - " + str(item[1]) + " sec\n"

    segments = find_contiguous_colors(colors)
    fig = plt.figure()
    fig.suptitle(sound + " total similarity: " + str(fn_prob[0][1]) + "\n" + message)

    print("\n\n" + message)

    start= 0
    for seg in segments:
        end = start + len(seg)
        l, = plt.gca().plot(x[start:end],y[start:end],lw=2,c=seg[0])
        start = end


def plot_prediction_matches(signal, match_plot, fn_prob, stamps):
    x = np.arange(len(signal))
    x_sec = np.arange(len(signal))/rate

    match_plot_final = []
    for k in range(len(match_plot)):
        i = match_plot[k]
        for j in range(step):
            match_plot_final.append(i)
    
    match_plot_final_c = []
    for p in match_plot_final:
        if(p == 1) and (fn_prob[0][1] >= fn_prob[0][0]):
            match_plot_final_c.append("green")
        elif(p == 1) and (fn_prob[0][1] < fn_prob[0][0]):
            match_plot_final_c.append("red")
        elif(p == 0):
            match_plot_final_c.append("blue")
    
    plot_multicolored_lines(x_sec,list(signal),match_plot_final_c, fn_prob, stamps)
    #labels = [str(item/rate) for item in x]
    #plt.xticks(x, labels)
    plt.show()


# In[ ]:


fn_prob, match_plot, wav, stamps = single_prediction(file_path)
print("True class similarity: " + str(fn_prob[0][1])) 
plot_prediction_matches(wav, match_plot, fn_prob, stamps)

