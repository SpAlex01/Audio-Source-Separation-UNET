import os
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define the model architecture
class AudioUNet(nn.Module):
    def __init__(self, input_channels, start_neurons):
        super(AudioUNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels, start_neurons, kernel_size=3, padding=1, stride=2
            ),
            nn.Conv2d(start_neurons, start_neurons, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                start_neurons, start_neurons * 2, kernel_size=3, padding=1, stride=2
            ),
            nn.Conv2d(start_neurons * 2, start_neurons * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                start_neurons * 2, start_neurons * 4, kernel_size=3, padding=1, stride=2
            ),
            nn.Conv2d(start_neurons * 4, start_neurons * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                start_neurons * 4,
                start_neurons * 8,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            nn.Conv2d(start_neurons * 8, start_neurons * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        self.convm = nn.Sequential(
            nn.Conv2d(
                start_neurons * 8,
                start_neurons * 16,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            nn.Conv2d(
                start_neurons * 16,
                start_neurons * 16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.deconv4 = nn.ConvTranspose2d(
            start_neurons * 16,
            start_neurons * 8,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.uconv4 = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(start_neurons * 16, start_neurons * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(start_neurons * 16, start_neurons * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.deconv3 = nn.ConvTranspose2d(
            start_neurons * 8,
            start_neurons * 4,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.uconv3 = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(start_neurons * 8, start_neurons * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(start_neurons * 8, start_neurons * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.deconv2 = nn.ConvTranspose2d(
            start_neurons * 4,
            start_neurons * 2,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.uconv2 = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(start_neurons * 4, start_neurons * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(start_neurons * 4, start_neurons * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.deconv1 = nn.ConvTranspose2d(
            start_neurons * 2,
            start_neurons,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.uconv1 = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(start_neurons * 2, start_neurons * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(start_neurons * 2, start_neurons, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.ConvTranspose2d(
            start_neurons,
            1,
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        x = torch.nn.functional.pad(x, (0, 22), "constant", 0)
        x = x[:, :, :-1, :]

        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)

        convm_out = self.convm(conv4_out)

        deconv4_out = self.deconv4(convm_out)
        uconv4_out = torch.cat((deconv4_out, conv4_out), dim=1)
        uconv4_out = self.uconv4(uconv4_out)

        deconv3_out = self.deconv3(uconv4_out)
        uconv3_out = torch.cat([deconv3_out, conv3_out], dim=1)
        uconv3_out = self.uconv3(uconv3_out)

        deconv2_out = self.deconv2(uconv3_out)
        uconv2_out = torch.cat([deconv2_out, conv2_out], dim=1)
        uconv2_out = self.uconv2(uconv2_out)

        deconv1_out = self.deconv1(uconv2_out)
        uconv1_out = torch.cat([deconv1_out, conv1_out], dim=1)
        uconv1_out = self.uconv1(uconv1_out)

        output = torch.sigmoid(self.output_layer(uconv1_out))

        output = output[:, :, :, :-22]

        output = torch.cat((output, output[:, :, -1, :].unsqueeze(2)), dim=2)

        return output

# Create an instance of the model
model = AudioUNet(input_channels=1, start_neurons=16)

# Load the trained model weights
checkpoint_path = 'models/checkpoint_epoch.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

# Load the noisy audio file
noisy_audio_file = 'F:\IRRI\combinate\mixed_audio_90.wav'
noisy_waveform, sample_rate = sf.read(noisy_audio_file, always_2d=True)
noisy_waveform = torch.from_numpy(noisy_waveform).to(torch.float32)

# Apply STFT to the noisy audio file
noisy_specgram = torch.stft(
    noisy_waveform.squeeze(),
    n_fft=1024,
    hop_length=512,
    win_length=1024,
    window=torch.hann_window(1024),
    return_complex=True,  # Ensure complex output for istft
)
noisy_magnitude = torch.sqrt(noisy_specgram.real ** 2 + noisy_specgram.imag ** 2)

# Forward pass through the model
with torch.no_grad():
    output_magnitude = model(noisy_magnitude.unsqueeze(0))

# Convert the output back to numpy array
output_magnitude = output_magnitude.squeeze().numpy()

# Convert the output_magnitude to a PyTorch tensor
output_magnitude = torch.from_numpy(output_magnitude)

# Reconstruct the denoised audio waveform
output_specgram = output_magnitude * torch.exp(1j * torch.angle(noisy_specgram))
denoised_waveform = torch.istft(output_specgram, n_fft=1024, hop_length=512, win_length=1024,
                                window=torch.hann_window(1024), length=len(noisy_waveform))

# Convert the denoised waveform to a numpy array
denoised_waveform = denoised_waveform.numpy()

# Save the denoised audio as a new WAV file
denoised_audio_file = 'denoised.wav'
sf.write(denoised_audio_file, denoised_waveform, sample_rate)

print(f"Denoised audio saved at: {denoised_audio_file}")
