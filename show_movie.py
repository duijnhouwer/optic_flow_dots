import tkinter as tk
from tkinter import filedialog
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def select_file():
    """Open a file dialog to select the tensor file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Ensure file dialog is on top
    file_path = filedialog.askopenfilename(initialdir=os.path.dirname(__file__)+'_data')
    root.destroy()  # Close the Tkinter root window after file selection
    return file_path

def load_tensor(file_path):
    """Load a PyTorch tensor from a file."""
    return torch.load(file_path)

def animate_movie(tensor):
    """Animate a movie stored in a PyTorch tensor."""
    fig, ax = plt.subplots()

    def update(frame_number):
        ax.clear()
        ax.imshow(tensor[0, frame_number], cmap='gray')
        ax.set_axis_off()

    ani = animation.FuncAnimation(fig, update, frames=range(tensor.size(1)), interval=50)
    plt.show()

def main():
    file_path = select_file()
    if file_path:
        tensor = load_tensor(file_path)
        animate_movie(tensor)
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()