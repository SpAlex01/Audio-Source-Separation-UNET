import os
import soundfile as sf
import numpy as np
import wavio
from scipy.signal import resample_poly

def process_flac(input_file, output_folder):
    try:
        # Load the FLAC file
        data, samplerate = sf.read(input_file)

        # Trim to 10 seconds
        if len(data) > 10 * samplerate:
            data = data[:10 * samplerate]

            # Convert to mono
            if len(data.shape) == 2:
                data = np.mean(data, axis=1)

            # Resample to 48kHz
            data_resampled = resample_poly(data, 48000, samplerate)

            # Ensure the length is exactly 480,000 samples
            target_length = 480000
            if len(data_resampled) < target_length:
                # Zero-pad if the audio is shorter than the target length
                data_resampled = np.pad(data_resampled, (0, target_length - len(data_resampled)))
            elif len(data_resampled) > target_length:
                # Trim if the audio is longer than the target length
                data_resampled = data_resampled[:target_length]

            # Get the file name (without extension) and create the output path with .wav extension
            file_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_folder, file_name + ".wav")

            # Save the processed file to the output folder using soundfile
            sf.write(output_file, data_resampled, 48000)

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def process_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.flac'):
                input_file = os.path.join(root, file)
                process_flac(input_file, output_folder)

def resample_folder(input_folder, output_folder, target_sr=48000):

    os.makedirs(output_folder, exist_ok=True)


    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"resampled_{filename}")

            # Resample audio
            resample_file(input_path, output_path, target_sr)

def resample_file(input_path, output_path, target_sr=48000):

    audio, sr = sf.read(input_path)

    # Convert to mono
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)

    # Resample audio using scipy
    resampled_audio = resample_poly(audio, target_sr, sr)

    # Ensure the length is exactly 480,000 samples
    target_length = 480000
    if len(resampled_audio) < target_length:
        # Zero-pad if the audio is shorter than the target length
        resampled_audio = np.pad(resampled_audio, (0, target_length - len(resampled_audio)))
    elif len(resampled_audio) > target_length:

        resampled_audio = resampled_audio[:target_length]

    # Save resampled audio
    sf.write(output_path, resampled_audio, target_sr)


if __name__ == "__main__":

    input_folder = "noise/train-noise1"


    output_folder = "noise/train-noise"

    resample_folder(input_folder, output_folder)
