import torch
from torchvision import transforms
from loguru import logger
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

class SuperResolutionInference:
    def __init__(self, model_path, model_class, device='cuda'):
        """
        Initialize the Super-Resolution inference class.

        Args:
            model_path (str): Path to the trained model checkpoint.
            model_class (torch.nn.Module): Model class (architecture).
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path, model_class)
        logger.info(f"Model loaded and moved to device: {self.device}")

    def load_model(self, model_path, model_class):
        """
        Load the model from the given checkpoint.

        Args:
            model_path (str): Path to the model checkpoint.
            model_class (torch.nn.Module): Model class to initialize.

        Returns:
            model: Loaded model ready for inference.
        """
        logger.info(f"Loading checkpoint from {model_path}")

        # Initialize model architecture
        model = model_class
        # Load checkpoint safely
        checkpoint = torch.load(model_path, map_location="cpu")
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])  # Lightning checkpoint
        else:
            model.load_state_dict(checkpoint)  # Standard state_dict checkpoint
        # Set the model to evaluation mode and move it to the device
        model.eval()
        model = model.to(self.device)
        return model

    def preprocess(self, lr_data, lr_mean, lr_std):
        """
        Preprocess low-resolution (LR) data for inference.

        Args:
            lr_data (xarray.DataArray): Low-resolution data to preprocess.
            lr_mean (float): Mean value for normalization.
            lr_std (float): Standard deviation for normalization.

        Returns:
            torch.Tensor: Preprocessed LR data ready for inference.
        """

        self.lr_mean = lr_mean
        self.lr_std = lr_std
        logger.info(f"Preprocessing data with mean={lr_mean}, std={lr_std}")

        # Define normalization transformation
        lr_transform = transforms.Normalize(mean=[lr_mean], std=[lr_std])

        # Convert LR data to tensor and reshape to (1, H, W)
        lr_tensor = torch.tensor(lr_data.values, dtype=torch.float32).unsqueeze(0)

        # Apply normalization and add batch dimension for inference (1, 1, H, W)
        lr_tensor = lr_transform(lr_tensor).unsqueeze(0)

        return lr_tensor

    def inference(self, lr_image):
        """
        Perform inference with the super-resolution model.

        Args:
            lr_image (torch.Tensor): Preprocessed LR image tensor (1, 1, H, W).

        Returns:
            sr_image (torch.Tensor): Super-resolved image tensor (1, 1, H, W).
        """
        # Move the LR image to the correct device
        lr_image = lr_image.to(self.device)
        # Disable gradient calculation for inference
        with torch.no_grad():
            sr_image = self.model(lr_image)
        return sr_image.cpu().squeeze().numpy()

    def visualize(self, lr_image, sr_image, lr_time, lon_min=5, lon_max=16.25, lat_min=47, lat_max=58.22):
        """
        Visualize the LR input and SR output images.

        Args:
            lr_image (xarray.DataArray): Low-resolution image for comparison.
            sr_image (np.ndarray): Super-resolved output image.
            lr_time (str): Timestamp for the LR image (used in plot titles).
            lon_min, lon_max, lat_min, lat_max: Coordinates for the plot extent.
        """
        sr_image = (sr_image * self.lr_std) + self.lr_mean

        v_min = min(lr_image.values.min(), sr_image.min() )
        v_max = max(lr_image.values.max(), sr_image.max() )
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.Mercator()})
        # Plot low-resolution data
        ax[0].coastlines()
        ax[0].add_feature(cf.BORDERS)
        ax[0].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax[0].imshow(lr_image.values, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max],
                     transform=ccrs.PlateCarree(), cmap="YlOrRd", vmin=v_min, vmax = v_max)
        ax[0].set_title(f"Low-Resolution: {lr_time}")


        ax[1].coastlines()
        ax[1].add_feature(cf.BORDERS)
        ax[1].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        hr = ax[1].imshow(sr_image, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max],
                     transform=ccrs.PlateCarree(), cmap="YlOrRd",vmin=v_min, vmax = v_max)
        ax[1].set_title(f"Super-Resolution: {lr_time}")
        cbar_hr = fig.colorbar(hr, ax=ax[1])
        cbar_hr.set_label(f"Temperature ({lr_image.units})")  # Add units to the colorbar
        # Add colorbar for the super-resolved data with the same units
        plt.tight_layout()
        plt.show()

        return fig