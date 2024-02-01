import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load an example audio file
audio_file_path = "voce/train-voce/61-70970-0004.wav"
y, sr = librosa.load(audio_file_path, sr=48000)  # Assuming 48 kHz sample rate

# Compute the spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

# Display the spectrogram for visualization
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.show()

# Print the shape of the spectrogram
print("Spectrogram shape:", D.shape)
