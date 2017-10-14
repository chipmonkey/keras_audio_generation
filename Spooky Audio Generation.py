
# coding: utf-8

# In[13]:


# Mutated from: https://gist.github.com/naotokui/12df40fa0ea315de53391ddc3e9dc0b9

# Spooky 1 - Scary Dark Ambient Music 1 Hour Of Best Ambient Horror Music by Noctilucant.mp3
# Spooky 2 - Creepy Wind.mp3
# Spooky 3 - Decomentarium_-_Dementia.mp3
# Spooky 4 - John Carpenters Halloween by Trent Reznor  Atticus Ross (Official Audio).mp3

import seaborn
import librosa
import numpy as np

audio_filename = './spooky2.wav'

sr = 8000
ally, _ = librosa.load(audio_filename, sr=sr, mono=True)
print(y.shape)



# In[15]:


# Shorten sample for testing with laptop RAM to 10%:
ylen = len(ally) // 10
print(ylen)
y = ally[40000:40000+ylen]

print(y.shape)

min_y = np.min(y)
max_y = np.max(y)

# normalize
y = (y - min_y) / (max_y - min_y)
print(y.dtype, min_y, max_y)

from IPython.display import Audio
Audio(y, rate=sr)


# In[16]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(30,5))
plt.plot(y[50000:100000].transpose())
plt.show()


# In[17]:


# Build a model in keras

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import RMSprop
# import tensorflow as tf

# so try to estimate next sample afte given (maxlen) samples
maxlen     = 128 # 128 / sr = 0.016 sec
nb_output = 256  # resolution - 8bit encoding
latent_dim = 128 

inputs = Input(shape=(maxlen, nb_output))
x = LSTM(latent_dim, return_sequences=True)(inputs)
x = Dropout(0.4)(x)
x = LSTM(latent_dim)(x)
x = Dropout(0.4)(x)
output = Dense(nb_output, activation='softmax')(x)
model = Model(inputs, output)

#optimizer = Adam(lr=0.005)
optimizer = RMSprop(lr=0.01) 
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[19]:


from tqdm import tqdm

# try to estimate next_sample (0 -255) based on 256 previous samples 
step = 5
next_sample = []
samples = []
for j in tqdm(range(0, y.shape[0] - maxlen, step)):
    seq = y[j: j + maxlen + 1]  
    seq_matrix = np.zeros((maxlen, nb_output), dtype=bool) 
    for i,s in enumerate(seq):
        sample_ = int(s * (nb_output - 1)) # 0-255
        if i < maxlen:
            seq_matrix[i, sample_] = True
        else:
            seq_vec = np.zeros(nb_output, dtype=bool)
            seq_vec[sample_] = True
            next_sample.append(seq_vec)
    samples.append(seq_matrix)
samples = np.array(samples, dtype=bool)
next_sample = np.array(next_sample, dtype=bool)
print(samples.shape, next_sample.shape)


# In[ ]:


from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
csv_logger = CSVLogger('training_audio.log')
escb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
checkpoint = ModelCheckpoint("models/audio-{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, period=2)

model.fit(samples, next_sample, shuffle=True, batch_size=256, verbose=1, #initial_epoch=50,
          validation_split=0.1, epochs=5, callbacks=[csv_logger, escb, checkpoint])

