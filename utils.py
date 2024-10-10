import os
import imageio
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm


def create_gif_from_images(trainer, output_name="training_progress.gif", duration=10):
    """
    Creates a GIF from PNG images saved in the current trainer log directory.

    Args:
        trainer (pl.Trainer): The Lightning Trainer object to get the log directory.
        output_name (str): The name of the output GIF file.
        duration (float): Duration (in seconds) for each frame in the GIF.
    """
    log_dir = trainer.log_dir  # Get the current logging directory
    log_dir = os.path.join(log_dir, "val_prediction")

    # Find all PNG files in the log directory
    images = [img for img in os.listdir(log_dir) if img.endswith(".png")]

    # Sort images by name (to ensure they are in correct order)
    images = natsorted(images)


    # Create full paths to images
    image_paths = [os.path.join(log_dir, img) for img in images]

    # Read and create GIF
    gif_path = os.path.join(log_dir, output_name)
    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
        for filename in image_paths:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"GIF saved at {gif_path}")


class ColorMappingGenerator:
    def __init__(self, lr_image, sr_result, num_colors=20, colormap="YlOrRd"):
        """
        Initialize the ColorMappingGenerator with image data and parameters.

        Parameters:
        - lr_image: Low-resolution image array (temperature values or similar)
        - sr_result: Super-resolution result array
        - num_colors: Number of discrete colors in the colormap
        - colormap: The colormap to use (default: 'YlOrRd')
        """
        self.num_colors = num_colors
        self.v_min = min(lr_image.min(), sr_result.min())  # Minimum value from both images
        self.v_max = max(lr_image.max(), sr_result.max())  # Maximum value from both images
        self.color_values = np.linspace(self.v_min, self.v_max, self.num_colors)  # Generate evenly spaced values
        self.cmap = plt.get_cmap(colormap)  # Get colormap
        self.norm = mpl.colors.Normalize(vmin=self.v_min, vmax=self.v_max)  # Normalize values
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)  # Colormap scalar

    def write_rgb_mapping_to_file(self, filename="color_mapping.txt"):
        """
        Write the RGB mapping to a text file.

        Parameters:
        - filename: The file to save the RGB mapping (default: 'color_mapping.txt')
        """
        with open(filename, 'w') as file:
            # Add special case for missing values (-9999) as black
            file.write(f"{-9999.0}, {0}, {0}, {0}\n")
            # Write RGB mappings for each value
            for value in self.color_values:
                rgba = self.scalarMap.to_rgba(value)  # Get RGBA color
                rgb = tuple(int(x * 255) for x in rgba[:3])  # Convert to 0-255 range (ignore alpha)
                file.write(f"{value:.2f} {rgb[0]}, {rgb[1]}, {rgb[2]}\n")
        print(f"Color mapping saved to {filename}.")

    def visualize_colorbar(self):
        """
        Visualize the colorbar corresponding to the colormap and value range.
        """
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        # Create and display the colorbar
        colorbar = mpl.colorbar.ColorbarBase(ax, cmap=self.cmap, norm=self.norm, orientation='horizontal')
        colorbar.set_label('Value')
        plt.show()