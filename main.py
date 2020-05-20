"""
Copyright (c) 2020, Heung Kit Leslie Chung
All rights reserved.
Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""

from music21 import *
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils, to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import time
import pygame
import base64
import os

#=========================Music player function================================
# This function is for playing tracks from file
def play_music(music_file, duration = 10):
    clock = pygame.time.Clock()
    pygame.init()
    try:
        pygame.mixer.music.load(music_file)
        print("Music file %s loaded!" % music_file)
    except pygame.error:
        print("File %s not found! (%s)" % (music_file, pygame.get_error()))
    pygame.mixer.music.play()
    time.sleep(duration)
    pygame.mixer.quit()

# This function is for playing midi score/part objects
def play_midi(midi, duration = 10):
    clock = pygame.time.Clock()
    pygame.init()
    try:
        temp = stream.Stream(midi)
        temp.write('midi', fp='temp.mid')
        temp_midi = 'temp.mid'
        pygame.mixer.music.load(temp_midi)
        print("Midi file %s loaded!" % midi)
    except pygame.error:
        print("Midi object %s not found! (%s)" % (midi, pygame.get_error()))
    pygame.mixer.music.play()
    time.sleep(duration)
    pygame.mixer.quit()
    if os.path.exists('temp.mid'):
        os.remove('temp.mid')
        
#=======================Plot model loss function===============================
# summarize history for accuracy
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('LSTM_modl_loss.png')
    plt.close('all')

#========================Load in our music tracks==============================
# We will load in all available tracks in Music folder and append them into a list
notes = []
duration = []
delta = []

for i, file in enumerate(glob.glob("Music/*.mid")):
    # Load in the midi file
    midi = converter.parse(file)[0]        
    # Get notes/chords and duration
    for e in midi.flat.notes:
        if isinstance(e, note.Note):
            notes.append(str(e.pitch))
            duration.append(float(e.duration.quarterLength))
        elif isinstance(e, chord.Chord):
            # return numerical representation of chord (normal order)
            notes.append('.'.join(str(n) for n in e.normalOrder))
            duration.append(float(e.duration.quarterLength))
    # Get the offset of each note and convert to deltas
    deltatemp = []
    [deltatemp.append(x['offsetSeconds']) for x in midi.flat.notes.secondsMap]
    deltatemp = [deltatemp[x] - deltatemp[y] for x, y in zip(range(1,len(deltatemp)),range(0, len(deltatemp)-1))]
    deltatemp.insert(0,0)
    delta.extend(deltatemp)
    
    print('')
    print("{} Loaded".format(file))

#==========================Transform music data================================
# Get all pitch names
pitches = sorted(set(item for item in notes))
# Get all duration variations
durations = sorted(set(duration))
# Get all offset variations
deltas = sorted(set(delta))
# Count number of different pitches, notes, durations, offsets
pitch_count = len(pitches)  
note_count = len(notes)
speed_count = len(durations)
delta_count = len(deltas)


# Use one-hot encoding for each note and create an array 
# First index the possible notes
note_dict = dict()
for i, notev in enumerate(pitches):
    note_dict[notev] = i
    
# Do the same for durations
dur_dict = dict()
for i, dur in enumerate(sorted(set(durations))):
    dur_dict[dur] = i
    
# Do the same for offsets
delta_dict = dict()
for i, deltav in enumerate(deltas):
    delta_dict[deltav] = i

# Now let's construct sequences. Taking each note and encoding it as a numpy array with a 1 in the position of the note it has
seq_len = 50
# Lets make a numpy array with the number of training examples, sequence length, and the length of the one-hot-encoding
num_seq = note_count - seq_len

input_notes = np.zeros((num_seq, seq_len, pitch_count))
output_notes = np.zeros((num_seq, pitch_count))

for i in range(0, num_seq):
    # Load in notes in chunks
    input_sequence = notes[i: i+seq_len]
    # Output note is the next note after the input sequence, i.e. the prediction
    output_note = notes[i+seq_len]
    
    for j, notev in enumerate(input_sequence):
        input_notes[i][j] = to_categorical(note_dict[notev], len(pitches))
        
    output_notes[i] = to_categorical(note_dict[output_note], len(pitches))

# Notes with duration and offsets added at the very end of each array
input_note_d = []
for i in range(0, num_seq):
    start = 0
    obj = np.hstack((input_notes[i], np.array(pd.get_dummies(duration))[start:start+50].reshape(50,speed_count)))
    obj = np.hstack((obj, np.array(pd.get_dummies(delta))[start:start+50].reshape(50, delta_count)))
    input_note_d.append(obj)
    start += 1
input_note_d = np.array(input_note_d)

output_note_d = []
for i in range(0, num_seq):
    start = seq_len # This is 50
    obj = np.hstack((output_notes[i], np.array(pd.get_dummies(duration))[start]))
    obj = np.hstack((obj, np.array(pd.get_dummies(delta))[start]))
    output_note_d.append(obj)
    start += 1
output_note_d = np.array(output_note_d)

#===============================LSTM model=====================================
def LSTM_block(output_size):
  model = Sequential()
  model.add(LSTM(128, return_sequences=True, input_shape=(seq_len, output_size)))
  model.add(Dropout(0.2))
  model.add(LSTM(128, return_sequences=False))
  model.add(Dropout(0.2))
  model.add(Dense(output_size))
  return model

model = LSTM(pitch_count+speed_count+delta_count)
top_input = Input(shape=input_note_d.shape[1:])
embedding = model(top_input)

note_outs = Dense(pitch_count, activation='softmax')(embedding)
duration_outs = Dense(speed_count, activation='softmax')(embedding)
delta_outs = Dense(delta_count, activation='softmax')(embedding)

comb_modl = Model(top_input, [note_outs, duration_outs, delta_outs])
comb_modl.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],
                  optimizer='rmsprop',metrics=['acc'])

plot_model(comb_modl, to_file='LSTM_model.png', expand_nested=True, show_shapes=True, show_layer_names=True)

#=============================Train LSTM model=================================
history = comb_modl.fit(input_note_d, [output_notes, np.array(pd.get_dummies(duration))[50:], 
                                       np.array(pd.get_dummies(delta))[50:]], 
                        batch_size=128, epochs=500)

plot_loss(history)

#===========================Use model to write song============================
# Make a dictionary going backwards (with index as key and the note as the value)
backward_dict = dict()
for notev in note_dict.keys():
    index = note_dict[notev]
    backward_dict[index] = notev

# Same for durations
backward_dur = dict()
for durv in dur_dict.keys():
    index = dur_dict[durv]
    backward_dur[index] = durv
    
# Same for deltas
backward_delta = dict()
for deltav in delta_dict.keys():
    index = delta_dict[deltav]
    backward_delta[index] = deltav

# Pick a random sequence from the input as a starting point for the prediction
n = np.random.randint(0, len(input_note_d)-1)
sequence = input_note_d[n]
start_sequence = sequence.reshape(1, seq_len, pitch_count+speed_count+delta_count)
output = []
dur = []
delts = []
# Generate song with 100 notes
for i in range(0, 200):
    newNote, durat, delt = comb_modl.predict(start_sequence, verbose=0)
    # Get the position with the highest probability for note
    index = np.argmax(newNote)
    encoded_note = to_categorical(index, pitch_count)
    output.append(encoded_note)
    # Do the same for duration
    index2 = np.argmax(durat)
    encoded_durat = to_categorical(index2, speed_count)
    dur.append(encoded_durat)
    # Do the same for delta
    index3 = np.argmax(delt)
    encoded_delts = to_categorical(index3, delta_count)
    delts.append(encoded_delts)
    
    sequence = start_sequence[0][1:]
    start_sequence = np.concatenate((sequence, 
                                     np.concatenate((encoded_note, encoded_durat, encoded_delts)).reshape(1, pitch_count+speed_count+delta_count)))
    start_sequence = start_sequence.reshape(1, seq_len, pitch_count+speed_count+delta_count)

finalNotes = [] 
for element in output:
    index = list(element).index(1)
    finalNotes.append(backward_dict[index])

finalDurations = []
for element in dur:
    index = list(element).index(1)
    finalDurations.append(backward_dur[index])

finalOffsets = []
for element in delts:
    index = list(element).index(1)
    finalOffsets.append(backward_delta[index])
    
offset = 0
output_notes = []

# Create note and chord objects based on the values generated by the model
for i, pattern in enumerate(finalNotes):
    # If pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(pitch.Pitch(int(current_note)), quarterLength=finalDurations[i])
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # If pattern is a note
    else:
        new_note = note.Note(pitch.Pitch(pattern), quarterLength=finalDurations[i])
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    # Increase offset each iteration so that notes do not stack
    offset += (finalOffsets[i] + 0.25)
    
#=============================Save song as MIDI================================
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='test_output.mid')