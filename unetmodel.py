import os

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Definirea funcției de construire a modelului pentru date audio
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


# Definirea clasei pentru setul de date
class AudioDataset(Dataset):
    def __init__(self, clean_files, noise_files):
        self.clean_files = clean_files
        self.noise_files = noise_files

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_file = self.clean_files[idx]
        noise_file = self.noise_files[idx]

        # Încarcă semnalul audio și aplică STFT
        clean_waveform, _ = sf.read(clean_file, always_2d=True)
        clean_waveform = torch.from_numpy(clean_waveform).to(torch.float32)

        clean_specgram = torch.stft(
            clean_waveform.squeeze(),
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            window=torch.hann_window(1024),
            return_complex=False,
        )
        clean_magnitude = torch.sqrt(
            clean_specgram[..., 0] ** 2 + clean_specgram[..., 1] ** 2
        )

        # Încarcă semnalul audio și aplică STFT pentru semnalul cu zgomot
        noise_waveform, _ = sf.read(noise_file, always_2d=True)
        noise_waveform = torch.from_numpy(noise_waveform).to(torch.float32)
        noise_specgram = torch.stft(
            noise_waveform.squeeze(),
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            window=torch.hann_window(1024),
            return_complex=False,
        )
        noise_magnitude = torch.sqrt(
            noise_specgram[..., 0] ** 2 + noise_specgram[..., 1] ** 2
        )

        return {"input": noise_magnitude, "target": clean_magnitude}


# Definirea funcției de pierdere și optimizatorului
criterion = nn.MSELoss()
input_channels = 1  # Pentru semnale audio monofonice; ajustează la 2 pentru stereo
start_neurons = 16  # Ajustează la necesitățile tale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioUNet(input_channels, start_neurons)
optimizer = optim.Adam(model.parameters(), lr=0.0008)

# Directorul unde se află fișierele audio pentru antrenare

clean_data_folder = "voce/train-voce"
clean_files = [
    os.path.join(clean_data_folder, file)
    for file in os.listdir(clean_data_folder)
    if file.endswith(".wav")
]

# Directorul unde se află fișierele cu zgomot
noise_data_folder = "combinate"
noise_files = [
    os.path.join(noise_data_folder, file)
    for file in os.listdir(noise_data_folder)
    if file.endswith(".wav")
]

# Definirea clasei pentru setul de date
dataset = AudioDataset(clean_files, noise_files)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Antrenarea rețelei
checkpoint_dir = 'models'
os.makedirs(checkpoint_dir, exist_ok=True)
num_epochs = 100
start_epoch = 0  # Change this if you want to start from a specific epoch
best_loss = float('inf')  # Initialize with a high value

# Attempt to load the latest checkpoint
latest_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch.pth')
if os.path.exists(latest_checkpoint_path):
    checkpoint = torch.load(latest_checkpoint_path)
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint.get('best_loss', float('inf'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Resuming training from epoch {start_epoch + 1}")

for epoch in range(start_epoch, num_epochs):
    for batch in dataloader:
        input_data = batch["input"]
        target_data = batch["target"]

        # Set gradients to zero
        optimizer.zero_grad()

        # Forward pass
        output = model(input_data)

        # Calculate the loss
        loss = criterion(output, target_data)

        # Backward pass and update weights
        loss.backward()
        optimizer.step()

    # Save checkpoint only if it's the best epoch so far
    if loss < best_loss:
        best_loss = loss.item()
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }, checkpoint_path)

        # Update the latest checkpoint file
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }, latest_checkpoint_path)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Best Loss: {best_loss}, Checkpoint saved at {checkpoint_dir}")

print("Training completed.")