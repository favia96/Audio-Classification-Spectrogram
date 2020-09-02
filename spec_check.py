from itertools import islice
from pathlib import Path
from IPython.display import Audio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.signal import get_window
# make sure to install all modules

plt.rcParams['figure.figsize'] = (12, 3)

DATA = Path('') #add the directory
AUDIO = DATA/'freesound' #put folder containing wav files
CSV = DATA/'freesound.csv' #file with labels in csv format

df = pd.read_csv(CSV)
# print(df.head(3))

row = df.iloc[1] # saxophone clip
filename = AUDIO / row.fname

# open the audio file
clip, sample_rate = librosa.load(filename, sr=None)

# print('Sample Rate   {} Hz'.format(sample_rate))
# print('Clip Length   {:3.2f} seconds'.format(len(clip)/sample_rate))

three_seconds = sample_rate * 3
clip = clip[:three_seconds]

timesteps = np.arange(len(clip)) / sample_rate  # in seconds

fig, ax = plt.subplots(2, figsize=(12, 5))
fig.subplots_adjust(hspace=0.5)

# plot the entire clip 
ax[0].plot(timesteps, clip)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('Raw Audio: {} ({} samples)'.format(row.label, len(clip)))


n_fft = 1024 # frame length 
start = 45000 # start at a part of the sound thats not silence.. 
x = clip[start:start+n_fft]

# mark location of frame in the entire signal
ax[0].axvline(start/sample_rate, c='r') 
ax[0].axvline((start+n_fft)/sample_rate, c='r')

# plot N samples 
ax[1].plot(x)
ax[1].set_xlabel('Samples')
ax[1].set_ylabel('Amplitude')
ax[1].set_title('Raw Audio: {} ({} samples)'.format(row.label, len(x)));
plt.show()
