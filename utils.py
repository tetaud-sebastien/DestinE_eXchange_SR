import os
import imageio
from natsort import natsorted

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