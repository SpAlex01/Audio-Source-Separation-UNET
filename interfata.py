import tkinter as tk
from tkinter import ttk, filedialog
import subprocess
import threading
import pygame
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Processor")

        # Set style for themed widgets
        self.style = ttk.Style()
        self.style.configure(
            "TButton",
            padding=10,
            relief="flat",
            background="#4CAF50",
            foreground="black",  # Set text color to black
            font=('Times New Roman', 12),  # Set font to Times New Roman, size 12
        )
        self.style.map(
            "TButton",
            foreground=[('active', 'white')],
            background=[('active', '#45a049')],
        )

        # Buttons and entry widgets
        self.insert_button = ttk.Button(master, text="Insert Noisy File", command=self.insert_noisy_file, style="TButton")
        self.insert_button.pack(pady=10)

        self.path_entry = ttk.Entry(master, width=50)
        self.path_entry.pack(pady=10)

        self.process_button = ttk.Button(master, text="Process", command=self.process_audio, style="TButton")
        self.process_button.pack(pady=10)

        self.play_button = ttk.Button(master, text="Play Audio File", command=self.play_audio, style="TButton")
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(master, text="Stop Audio", command=self.stop_audio, style="TButton")
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.show_spectrogram_button = ttk.Button(master, text="Show Spectrogram", command=self.show_spectrogram, style="TButton")
        self.show_spectrogram_button.pack(pady=10)

        # Variable to control audio playback thread
        self.audio_thread = None

        # Add animation effect to buttons
        self.animate_buttons()

    def insert_noisy_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, file_path)

    def process_audio(self):
        file_path = self.path_entry.get()
        if file_path:
            # Call the utilizare.py script using subprocess
            subprocess.run(["python", "utilizare.py", file_path])
            print("Processing complete!")

    def play_audio(self):
        file_path = self.path_entry.get()
        if file_path:
            # Start a new thread for audio playback
            self.audio_thread = threading.Thread(target=self.play_audio_thread, args=(file_path,))
            self.audio_thread.start()

    def play_audio_thread(self, file_path):
        pygame.mixer.init()
        pygame.mixer.music.load('denoised.wav')
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def stop_audio(self):
        # Stop the audio playback
        pygame.mixer.music.stop()
        pygame.mixer.quit()

    def show_spectrogram(self):
        # Call the plot.py script using subprocess
        subprocess.run(["python", "plot.py"])
        print("Plotting complete!")

    def generate_spectrogram(self, file_path):
        # Add your spectrogram generation logic here using matplotlib or any other library
        # Update this function according to your requirements
        pass

    def show_spectrogram_window(self, spectrogram):
        # Create a new window for displaying the spectrogram
        spectrogram_window = tk.Toplevel(self.master)
        spectrogram_window.title("Spectrogram")

        # Embed the matplotlib figure in the Tkinter window
        figure = plt.Figure(figsize=(6, 4), dpi=100)
        subplot = figure.add_subplot(111)
        subplot.imshow(spectrogram, cmap='viridis', aspect='auto')

        canvas = FigureCanvasTkAgg(figure, spectrogram_window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

    def animate_buttons(self):
        # Simple animation effect for buttons
        def toggle_color(button, color1, color2, delay):
            current_color = self.style.lookup(button.winfo_name(), 'background')
            next_color = color2 if current_color == color1 else color1
            self.style.configure(button.winfo_name(), background=next_color)
            self.master.after(delay, toggle_color, button, color1, color2, delay)

        for button in [self.insert_button, self.process_button, self.play_button, self.stop_button, self.show_spectrogram_button]:
            toggle_color(button, '#4CAF50', '#45a049', 500)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioProcessorApp(root)
    root.mainloop()
