from music21 import *
import glob
import numpy as np
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

#========================Load in our music tracks==============================
# We will load in all available tracks in Music folder and append them into a list
notes = []
duration = []

for i, file in enumerate(glob.glob("Music/*.mid")):
    
    midi = converter.parse(file)[0]        
        
    for e in midi.flat.notes:
        if isinstance(e, note.Note):
            notes.append(str(e.pitch))
            duration.append(float(e.duration.quarterLength))
        elif isinstance(e, chord.Chord):
            # return numerical representation of chord (normal order)
            notes.append('.'.join(str(n) for n in e.normalOrder))
            duration.append(float(e.duration.quarterLength))
    print('')
    print("{} Loaded".format(file))

#==========================Transform music data================================
# Get all pitch names
pitches = sorted(set(item for item in notes))
# Count number of different pitches
pitch_count = len(pitches)  
note_count = len(notes)

# Use one-hot encoding for each note and create an array 
# First index the possible notes
note_dict = dict()
for i, notev in enumerate(pitches):
    note_dict[notev] = i

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

# Notes with duration added at the very end of each array
input_note_d = []
for i in range(0, num_seq):
    start = 0
    input_note_d.append(np.hstack((input_notes[i], np.array(duration)[start:start+50].reshape(50,1))))
    start += 1
input_note_d = np.array(input_note_d)

output_note_d = []
for i in range(0, num_seq):
    start = seq_len # This is 50
    output_note_d.append(np.hstack((output_notes[i], duration[start])))
    start += 1
output_note_d = np.array(output_note_d)

#===============================LSTM model=====================================
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, pitch_count+1)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(pitch_count+1))

top_input = Input(shape=input_note_d.shape[1:])
embedding = model(top_input)

note_outs = Dense(pitch_count, activation='softmax')(embedding)
duration_outs = Dense(1, activation='relu')(embedding)

#note_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
#duration_modl.compile(loss='mse', optimizer='rmsprop',metrics=['acc'])

comb_modl = Model(top_input, [note_outs, duration_outs])
comb_modl.compile(loss=['categorical_crossentropy', 'mse'],optimizer='rmsprop',metrics=['acc'])

plot_model(comb_modl, to_file='LSTM_model.png' ,expand_nested=True, show_shapes=True, show_layer_names=True)

#=============================Train LSTM model=================================
#history = model.fit(input_note_d, output_note_d, batch_size=128, epochs=200)

history = comb_modl.fit(input_note_d, [output_notes, np.array(duration[50:]).reshape(num_seq, 1)], 
                        batch_size=128, epochs=200)

#===========================Use model to write song============================
# Make a dictionary going backwards (with index as key and the note as the value)
backward_dict = dict()
for notev in note_dict.keys():
    index = note_dict[notev]
    backward_dict[index] = notev

# Pick a random sequence from the input as a starting point for the prediction
n = np.random.randint(0, len(input_note_d)-1)
sequence = input_note_d[n]
start_sequence = sequence.reshape(1, seq_len, pitch_count+1)
output = []
dur = []
# Generate song with 100 notes
for i in range(0, 100):
    newNote, duration = comb_modl.predict(start_sequence, verbose=0)
    # Get the position with the highest probability
    index = np.argmax(newNote)
    encoded_note = to_categorical(index, pitch_count)
    output.append(encoded_note)
    dur.append(duration)
    sequence = start_sequence[0][1:]
    start_sequence = np.concatenate((sequence, 
                                     np.concatenate((encoded_note, duration.reshape(1,))).reshape(1, pitch_count+1)))
    start_sequence = start_sequence.reshape(1, sequence_length, pitch_count+1)
    
finalNotes = [] 
for element in output:
    index = list(element).index(1)
    finalNotes.append(backward_dict[index])
    
offset = 0
output_notes = []

# Create note and chord objects based on the values generated by the model
for i, pattern in enumerate(finalNotes):
    # If pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(pitch.Pitch(int(current_note)), quarterLength=float(dur[i]))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # If pattern is a note
    else:
        new_note = note.Note(pitch.Pitch(pattern), quarterLength=float(dur[i]))
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    # Increase offset each iteration so that notes do not stack
    offset += 0.5
    
#=============================Save song as MIDI================================
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='test_output.mid')