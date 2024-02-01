import os
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import csv
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F






class DatasetPersonalizat(Dataset):
    # constructor
    def __init__(self, csv_file_path, speech_files_directory, noise_files_directory, output_folder, seed=None,transform=None):
        self.df = pd.read_csv(csv_file_path)

        print("Dataset initialized.")
        self.speech_files_directory = speech_files_directory
        self.noise_files_directory = noise_files_directory
        self.output_folder = output_folder
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return len(self.df)

    # get files randomly with a seed
    def get_random_files(self, index):


        rng = np.random.RandomState(seed)

        speech_file_path = os.path.join(self.speech_files_directory, rng.choice(self.df['TrainVoce']))
        noise_file_path = os.path.join(self.noise_files_directory, rng.choice(self.df['TrainNoise']))

        return speech_file_path, noise_file_path

    def __getitem__(self, index):
        print(f"Getting item for index: {index}")
        speech_file_path = self.df.loc[index, 'TrainVoce']
        print(f"Speech file path: {speech_file_path}")
        speech_audio, sr = sf.read(speech_file_path)

        # Select random audio file for noise
        noise_file_path = self.df.loc[index, 'TrainNoise']
        print(f"Noise file path: {noise_file_path}")
        noise_audio, _ = sf.read(noise_file_path)
        if noise_audio.ndim > 1:
            noise_audio = noise_audio[:, 0]

        speech_audio = np.append(speech_audio[0], speech_audio[1:] - 0.97 * speech_audio[:-1])

        # normalization of speech to -30dB
        target_db = -30.0
        rmsspeech = np.sqrt(np.mean(np.square(speech_audio)))
        current_db = 20 * np.log10(rmsspeech)
        adjustment_factor = 10 ** ((target_db - current_db) / 20)
        speech_audio = speech_audio * adjustment_factor

        # normalization of noise to random target between -40dB and -30dB

        target_db_low = -40.0
        target_db_high = -30.0
        seed=25
        rng = np.random.RandomState(seed)
        random_target_db = rng.uniform(target_db_low, target_db_high)
        rmsnoise = np.sqrt(np.mean(np.square(noise_audio)))
        current_db = 20 * np.log10(rmsnoise)
        adjustment_factor = 10 ** ((random_target_db - current_db) / 20)
        noise_audio = noise_audio * adjustment_factor

        mixed_audio = speech_audio + noise_audio

        speech_audio = speech_audio.astype(np.float64)
        noise_audio = noise_audio.astype(np.float64)
        mixed_audio = mixed_audio.astype(np.float64)

        # Convert numpy arrays to torch tensors
        speech_audio = torch.from_numpy(speech_audio).double()
        noise_audio = torch.from_numpy(noise_audio).double()
        mixed_audio = torch.from_numpy(mixed_audio).double()


        if self.transform:
            mixed_audio = self.transform(mixed_audio)

        output_filename = f"mixed_audio_{index}.wav"
        output_filepath = os.path.join(self.output_folder, output_filename)

        # Debug prints
        print(f"Output folder: {self.output_folder}")
        print(f"Output filepath: {output_filepath}")

        # convert to numpy array to write to .wav
        sf.write(output_filepath, mixed_audio.reshape(-1, 1), sr)

        # debug
        if os.path.exists(output_filepath):
            print(f"File written successfully to: {output_filepath}")
        else:
            print(f"Error: File not written to {output_filepath}")

        return {
            'mixed_audio': mixed_audio,
            'speech_audio': speech_audio,
            'index': index
        }

def create_csv(csv_file_path, speech_files_directory, noise_files_directory, num_entries):
    # Get lists of all audio files in the specified directories
    speech_files = [f for f in os.listdir(speech_files_directory) if f.endswith('.wav')]
    noise_files = [f for f in os.listdir(noise_files_directory) if f.endswith('.wav')]

    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['TrainVoce', 'TrainNoise']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header to the CSV file
        writer.writeheader()

        # Generate random pairs of speech and noise files
        for _ in range(num_entries):
            speech_file = random.choice(speech_files)
            noise_file = random.choice(noise_files)

            # Write the pair to the CSV file
            writer.writerow({'TrainVoce': os.path.join(speech_files_directory, speech_file),
                             'TrainNoise': os.path.join(noise_files_directory, noise_file)})








if __name__ == "__main__":



    csv_file_path = "combinat.csv"
    speech_files_directory = "voce/train-voce"
    noise_files_directory = "noise/train-noise"
    num_entries=400
    output_folder = 'combinate'
    num_epochs=10
    #create_csv(csv_file_path, speech_files_directory,noise_files_directory,num_entries)
    seed=25

    # Assuming you have a Dataset class named DatasetPersonalizat
    dataset = DatasetPersonalizat(csv_file_path, speech_files_directory, noise_files_directory, output_folder)

    # Hyperparameters
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 10

    # DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    # Model (move to CPU)
    simple_model =UNet(in_channels=1025, out_channels=1)

    # simple_model = simple_model.cuda()  # Remove this line to keep it on CPU

    # Assuming you have a Dataset class named DatasetPersonalizat
    # Initialize the dataset, dataloader, loss function, and optimizer as in your code
    accumulation_steps = 4

    # Training loop
    for epoch in range(num_epochs):
        simple_model.train()
        total_loss = 0.0

        for batch in data_loader:
            mixed_audio = batch['mixed_audio'].unsqueeze(1)  # Add channel dimension
            voice_audio = batch['speech_audio'].unsqueeze(1)  # Add channel dimension

            # Forward pass
            separated_audio = simple_model(mixed_audio)

            # Compute the loss
            loss = criterion(separated_audio, voice_audio)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                # Update weights every accumulation_steps batches
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Save the trained simplified model
    torch.save(simple_model.state_dict(), 'simple_unet_model.pth')
