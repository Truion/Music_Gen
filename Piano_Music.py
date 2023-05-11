import numpy as np
import pickle
import streamlit as st

import os
import sys
import random
sys.path.append('C:/Users/Midrian/Documents/Misc/Music_generate/Music-Generation/midi')
import IPython.display as ipd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
import torch
import torch.utils.data as data

from Music_gen_utils import train_model, get_triangular_lr, NotesGenerationDataset

from midi_utils import midiread, midiwrite
from matplotlib import pyplot as plt
# import skimage.io as io
from IPython.display import FileLink
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#############################
import numpy as np
import torch
import torch.utils.data as data


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, n_layers=2):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_layers = n_layers
        
        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        
        self.bn = nn.BatchNorm1d(hidden_size)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        
        self.logits_fc = nn.Linear(hidden_size, num_classes)
    
    
    def forward(self, input_sequences, input_sequences_lengths, hidden=None):
        batch_size = input_sequences.shape[1]

        notes_encoded = self.notes_encoder(input_sequences)
        
        notes_encoded_rolled = notes_encoded.permute(1,2,0).contiguous()
        notes_encoded_norm = self.bn(notes_encoded_rolled)
        
        notes_encoded_norm_drop = nn.Dropout(0.25)(notes_encoded_norm)
        notes_encoded_complete = notes_encoded_norm_drop.permute(2,0,1)
        
        # Here we run rnns only on non-padded regions of the batch
        packed = torch.nn.utils.rnn.pack_padded_sequence(notes_encoded_complete, input_sequences_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        
        # Here we unpack sequence(back to padded)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        
        outputs_norm = self.bn(outputs.permute(1,2,0).contiguous())
        outputs_drop = nn.Dropout(0.1)(outputs_norm)
        logits = self.logits_fc(outputs_drop.permute(2,0,1))
        logits = logits.transpose(0, 1).contiguous()
        
        neg_logits = (1 - logits)
        
        # Since the BCE loss doesn't support masking,crossentropy is used
        binary_logits = torch.stack((logits, neg_logits), dim=3).contiguous()
        logits_flatten = binary_logits.view(-1, 2)
        return logits_flatten, hidden

def sample_from_piano_rnn(rnn, sample_length=4, temperature=1, starting_sequence=None):

    if starting_sequence is None:
                
        current_sequence_input = torch.zeros(1, 1, 88)
        current_sequence_input[0, 0, 40] = 1
        current_sequence_input[0, 0, 50] = 0
        current_sequence_input[0, 0, 56] = 0
        current_sequence_input = Variable(current_sequence_input.cuda())
    else:
        
        current_sequence_input = torch.zeros(1, 1, 88)
        current_sequence_input[0, 0, starting_sequence[0]] = 1
        current_sequence_input[0, 0, starting_sequence[1]] = 0
        current_sequence_input[0, 0, starting_sequence[2]] = 0
        current_sequence_input = Variable(current_sequence_input.cuda())
        # current_sequence_input = starting_sequence
        
    final_output_sequence = [current_sequence_input.data.squeeze(1)]

    hidden = None

    for i in range(sample_length):

        output, hidden = rnn(current_sequence_input, [1], hidden)

        probabilities = nn.functional.softmax(output.div(temperature), dim=1)

        current_sequence_input = torch.multinomial(probabilities.data, 1).squeeze().unsqueeze(0).unsqueeze(1)

        current_sequence_input = Variable(current_sequence_input.float())

        final_output_sequence.append(current_sequence_input.data.squeeze(1))

    sampled_sequence = torch.cat(final_output_sequence, dim=0).cpu().numpy()
    
    return sampled_sequence

# loading the saved model
rnn = torch.load('C:/Users/Midrian/Documents/Misc/Music_generate/Music-Generation/notebooks/Model_MUSIC.pth')

def main():
    attempt = 0
    st.title('Piano Melody Generation')
    st.markdown('This is a simple web application that uses a Recurrent Neural Network to generate a piano melody.')
    
    temperature = st.slider('Temperature', min_value=0.1, max_value=100.0, value=0.5, step=0.1)
    start_seq = st.text_input('Starting Sequence (comma separated list of integers)', '60,64,67')
    
    start_seq = [int(s) for s in start_seq.split(',')]
    
    # rnn.load_state_dict(torch.load('C:/Users/Midrian/Documents/Misc/Music_generate/Music-Generation/notebooks/music_model_padfront_regularized.pth'))
    
    if st.button('Generate'):
        # Run the RNN model on the starting sequence
        generated_notes = sample_from_piano_rnn (rnn, sample_length=200, temperature=temperature, starting_sequence=start_seq).transpose()
        
        # Save the generated melody as a MIDI file
        midi_filename = 'C:/Users/Midrian/Documents/Misc/Music_generate/saves/Music_{}.mid'.format(attempt)
        midiwrite(midi_filename, generated_notes, dt=0.3)
        attempt += 1
        
        # Display the generated melody as a plot and as a MIDI file download link
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.imshow(generated_notes.T, origin='lower', cmap='gray_r', aspect='auto')
        ax.set_xlabel('Time')
        ax.set_ylabel('Pitch')
        st.pyplot(fig)
        
        # Add a play button to play the generated MIDI file
        audio_data = open(midi_filename, 'rb').read()
        st.audio(audio_data, format='audio/midi')
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
  
    
  