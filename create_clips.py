import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import norm
import scipy.stats as stats
import seaborn as sns
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from IPython.display import Video
from pandas_profiling import ProfileReport
from helper import *

df = pd.read_csv('./videos/train.csv',low_memory=False)
# convert Nan to empty list
df[['event_attributes']] = df[['event_attributes']].fillna('')

df_throwin = df[df["event"] == "throwin"].reset_index()
df_play = df[df["event"] == "play"].reset_index()
df_challenge = df[df["event"] == "challenge"].reset_index()
events = []
paths = []

# create throwins clips from the df_throwin
for i in range(len(df_throwin)):
    video,event,path = create_video(df_throwin.iloc[i],'throwin')
    events.append(event)
    paths.append(path)
    

# create plays clips from the df_play
for i in range(len(df_play)):
    video,event,path =create_video(df_play.iloc[i],'play',before=2, after=2)
    events.append(event)
    paths.append(path)
    

# create challenges clips from the df_challenge
for i in range(len(df_challenge)):
    video,event,path =create_video(df_challenge.iloc[i], 'challenge',before=3, after=3)
    events.append(event)
    paths.append(path)


dict_df = {
    'event': events,
    'path': paths,
}
df = pd.DataFrame(dict_df)
df.to_csv('clips_events.csv', index = False, encoding='utf-8')

#video,event,path = create_video(df_play.iloc[0], 'challenge',before=3, after=3) 
#print(event.replace('[','').replace(']','').replace("'",''),path)