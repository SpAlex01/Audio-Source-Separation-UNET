import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch


denoised_audio_file = 'denoised.wav'
denoised_waveform, sample_rate = sf.read(denoised_audio_file, always_2d=True)
denoised_waveform = torch.from_numpy(denoised_waveform).to(torch.float32)

# Apply STFT to the denoised audio file
denoised_specgram = torch.stft(
    denoised_waveform.squeeze(),
    n_fft=1024,
    hop_length=512,
    win_length=1024,
    window=torch.hann_window(1024),
    return_complex=False,
)
denoised_magnitude = torch.sqrt(
    denoised_specgram[..., 0] ** 2 + denoised_specgram[..., 1] ** 2
)

# Convert the denoised magnitude to a NumPy array
denoised_magnitude_np = denoised_magnitude.numpy()

# Display the spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(
    10 * np.log10(denoised_magnitude_np),
    aspect='auto',
    origin='lower',
    cmap='viridis',
)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of Denoised Audio')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()