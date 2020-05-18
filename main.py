from music21 import converter, instrument, note, chord, midi, stream, pitch
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
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
notes = []

for i, file in enumerate(glob.glob("Music/*.mid")):
    midi = converter.parse(file)
    # Using first track in each file
    midi = midi[0]
    notes_to_parse = None
        
    # Parse the midi file by notes
    notes_to_parse = midi.flat.notes
        
    for e in tqdm(notes_to_parse):
        if isinstance(e, note.Note):
            notes.append(str(e.pitch))
        elif isinstance(e, chord.Chord):
            # return numerical representation of chord (normal order)
            notes.append('.'.join(str(n) for n in e.normalOrder))
    print('')
    print("Song {} Loaded".format(i+1))
                
#==========================Transform music data================================
# Get all pitch names
pitches = sorted(set(item for item in notes))
# Count number of different pitches
pitch_count = len(pitches)  
note_count = len(notes)

# Let's use One Hot Encoding for each of the notes and create an array as such of sequences. 
#Let's first assign an index to each of the possible notes
note_dict = dict()
for i, notev in enumerate(pitches):
    note_dict[notev] = i

# Now let's construct sequences. Taking each note and encoding it as a numpy array with a 1 in the position of the note it has
sequence_length = 50
# Lets make a numpy array with the number of training examples, sequence length, and the length of the one-hot-encoding
num_training = note_count - sequence_length

input_notes = np.zeros((num_training, sequence_length, pitch_count))
output_notes = np.zeros((num_training, pitch_count))

for i in range(0, num_training):
    # Here, i is the training example, j is the note in the sequence for a specific training example
    input_sequence = notes[i: i+sequence_length]
    output_note = notes[i+sequence_length]
    for j, notev in enumerate(input_sequence):
        input_notes[i][j][note_dict[notev]] = 1
    output_notes[i][note_dict[output_note]] = 1

#===============================LSTM model=====================================
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, pitch_count)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(pitch_count))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

plot_model(model, to_file='LSTM_model.png' ,expand_nested=True, show_shapes=True, show_layer_names=True)

#=============================Train LSTM model=================================
history = model.fit(input_notes, output_notes, batch_size=128, nb_epoch=200)

#===========================Use model to write song============================
# Make a dictionary going backwards (with index as key and the note as the value)
backward_dict = dict()
for notev in note_dict.keys():
    index = note_dict[notev]
    backward_dict[index] = notev

# Pick a random sequence from the input as a starting point for the prediction
n = np.random.randint(0, len(input_notes)-1)
sequence = input_notes[n]
start_sequence = sequence.reshape(1, sequence_length, vocab_length)
output = []

# Generate song with 100 notes
for i in range(0, 100):
    newNote = model.predict(start_sequence, verbose=0)
    # Get the position with the highest probability
    index = np.argmax(newNote)
    encoded_note = np.zeros((vocab_length))
    encoded_note[index] = 1
    output.append(encoded_note)
    sequence = start_sequence[0][1:]
    start_sequence = np.concatenate((sequence, encoded_note.reshape(1, vocab_length)))
    start_sequence = start_sequence.reshape(1, sequence_length, vocab_length)
    
finalNotes = [] 
for element in output:
    index = list(element).index(1)
    finalNotes.append(backward_dict[index])
    
offset = 0
output_notes = []

# Create note and chord objects based on the values generated by the model
for pattern in finalNotes:
    # If pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(pitch.Pitch(int(current_note)))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # If pattern is a note
    else:
        new_note = note.Note(pitch.Pitch(pattern))
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    # Increase offset each iteration so that notes do not stack
    offset += 0.5

#=============================Save song as MIDI================================
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='test_output.mid')