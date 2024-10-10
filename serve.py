import datetime
import gradio as gr
from gradio_calendar import Calendar
import xarray as xr
import numpy as np
import subprocess
import rioxarray  # Ensure rioxarray is imported for spatial data handling
from model import SRResNet
from super_resolution_inference import SuperResolutionInference
from utils import ColorMappingGenerator


# Constants for preprocessing
LR_MEAN = 10.007193565368652
LR_STD = 4.303609371185303
HR_MEAN = 10.094209671020508
HR_STD = 4.23423957824707
SCALE_FACTOR_LATITUDE = 8
MODEL_PATH = "/home/ubuntu/project/DestinE_eXchange_SR/lightning_logs/version_47/checkpoints/best-val-ssim-epoch=96-val_ssim=0.47.pt"


class GradioInference:
    """
    Class to handle the inference for super-resolution and generation of COG files
    for both low-resolution and super-resolved temperature data.
    """
    def __init__(self):
        """
        Initialize the GradioInference class. Load the dataset and initialize necessary parameters.
        """
        self.model_path = MODEL_PATH
        self.scale_factor_latitude = SCALE_FACTOR_LATITUDE

        # Load the dataset
        data = xr.open_dataset(
            "https://cacheb.dcms.destine.eu/d1-climate-dt/ScenarioMIP-SSP3-7.0-IFS-NEMO-0001-standard-sfc-v0.zarr",
            engine="zarr",
            storage_options={"client_kwargs": {"trust_env": "true"}},
            chunks={}
        )

        # Convert temperature to Celsius and extract required region
        t2m_lr = data.t2m.astype("float32") - 273.15  # Convert to Celsius
        t2m_lr.attrs["units"] = "C"
        self.lr = t2m_lr.sel(
            **{"latitude": slice(47, 58.22), "longitude": slice(5, 16.25)}
        )
        self.sr_result = None
        self.lr_image = None

    def run_inference_on_date(self, selected_date):
        """
        Run super-resolution inference for the selected date.

        Args:
            selected_date (datetime): The date to use for selecting the low-resolution data.

        Returns:
            Plot: Plot showing low-resolution and super-resolved images.
        """
        self.current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        model = SRResNet(
            large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=8
        )
        sr = SuperResolutionInference(model_path=self.model_path, model_class=model)
        # Convert the selected date to the appropriate format
        formatted_str = datetime.datetime.fromtimestamp(selected_date).strftime('%Y-%m-%dT%H:%M')
        # Currently hardcoded for demonstration
        self.lr_image = self.lr.sel(time=formatted_str, method='nearest')
        # Preprocess and perform inference
        preprocessed_image = sr.preprocess(self.lr_image, lr_mean=LR_MEAN, lr_std=LR_STD)
        sr_result = sr.inference(preprocessed_image)

        # Postprocess the result and visualize
        self.sr_result = sr.postprocessing(sr_result, HR_MEAN, HR_STD)
        fig = sr.visualize(lr_image=self.lr_image, sr_image=self.sr_result,
                           lr_time=self.lr_image.time.values
        )

        return fig

    def generate_cog_file(self):
        """
        Generate a COG file based on the super-resolved image and its latitude/longitude.

        Returns:
            str: Success or error message.
        """
        color_mapping_gen = ColorMappingGenerator(
            lr_image=self.lr_image.values, sr_result=self.sr_result, num_colors=20
        )
        # Write the RGB mapping to a text file
        color_mapping_gen.write_rgb_mapping_to_file("color_mapping.txt")

        try:
            if self.lr_image is None or self.sr_result is None:
                raise ValueError("Run inference before generating the COG file.")

            # Generate latitude and longitude
            latitudes = np.linspace(
                self.lr_image.latitude.min(), self.lr_image.latitude.max(),
                int(self.lr_image.shape[0] * self.scale_factor_latitude)
            )
            longitudes = np.linspace(
                self.lr_image.longitude.min(), self.lr_image.longitude.max(),
                int(self.lr_image.shape[1] * self.scale_factor_latitude)
            )

            # Create xarray Dataset for super-resolved data
            ds = xr.Dataset(
                data_vars={"t2m": (["latitude", "longitude"], self.sr_result)},
                coords={"latitude": latitudes, "longitude": longitudes, "time": self.lr_image.time},
                attrs={"description": "Super-resolved 2-meter temperature"}
            )
            var = ds['t2m']
            # Convert to proper CRS and handle missing values
            var = var.rename({'latitude': 'y', 'longitude': 'x'})
            var.rio.write_crs("EPSG:4326", inplace=True)
            var_filled = var.fillna(-9999)  # Replace NaNs with -9999
            # Save the super-resolved image as a TIF file
            tif_filename = 'tif_filename.tif'
            var_filled.rio.to_raster(tif_filename)
            # Convert to VRT and apply color relief
            vrt_filename = 'vrt_filename.vrt'
            output_vrt_filename = 'output_vrt_filename.vrt'
            subprocess.run(f"gdal_translate -of VRT {tif_filename} {vrt_filename}", shell=True, check=True)
            subprocess.run(f"gdaldem color-relief {vrt_filename} color_mapping.txt {output_vrt_filename}", shell=True, check=True)
            # Convert VRT to COG
            output_cog_filename = f"{self.current_time}_hr_output_cog_filename.tif"
            subprocess.run(f"gdal_translate -of COG {output_vrt_filename} {output_cog_filename}", shell=True, check=True)
            subprocess.run(f"rm -fr {tif_filename} {vrt_filename} {output_vrt_filename}", shell=True, check=True)

            return output_cog_filename  # Return the COG file path to be downloadable

        except subprocess.CalledProcessError as e:
            return f"Error occurred: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def generate_lr_image_cog_file(self):
        """
        Generate a COG file based on the low-resolution image and its latitude/longitude.

        Returns:
            str: Success or error message.
        """
        try:
            if self.lr_image is None:
                raise ValueError("Run inference before generating the COG file.")
            # Create xarray Dataset for the low-resolution image
            ds_lr = xr.Dataset(
                data_vars={"t2m": (["latitude", "longitude"], self.lr_image.values)},
                coords={"latitude": self.lr_image.latitude, "longitude": self.lr_image.longitude, "time": self.lr_image.time},
                attrs={"description": "Low-resolution 2-meter temperature"}
            )
            var_lr = ds_lr['t2m']
            # Convert to proper CRS and handle NaN values
            var_lr = var_lr.rename({'latitude': 'y', 'longitude': 'x'})
            var_lr.rio.write_crs("EPSG:4326", inplace=True)
            var_lr_filled = var_lr.fillna(-9999)  # Replace NaNs with -9999
            # Save the low-resolution data as a TIF file
            tif_lr_filename = 'lr_tif_filename.tif'
            var_lr_filled.rio.to_raster(tif_lr_filename)
            # Convert to VRT and apply color relief (Optional, depends on your needs)
            vrt_lr_filename = 'lr_vrt_filename.vrt'
            output_vrt_lr_filename = 'lr_output_vrt_filename.vrt'
            subprocess.run(f"gdal_translate -of VRT {tif_lr_filename} {vrt_lr_filename}", shell=True, check=True)
            subprocess.run(f"gdaldem color-relief {vrt_lr_filename} color_mapping.txt {output_vrt_lr_filename}", shell=True, check=True)
            # Convert VRT to COG
            output_cog_lr_filename = f"{self.current_time}_lr_output_cog_filename.tif"
            subprocess.run(f"gdal_translate -of COG {output_vrt_lr_filename} {output_cog_lr_filename}", shell=True, check=True)
            subprocess.run(f"rm -fr {vrt_lr_filename} {output_vrt_lr_filename} {tif_lr_filename}", shell=True, check=True)
            return output_cog_lr_filename  # Return the COG file path to be downloadable

        except subprocess.CalledProcessError as e:
            return f"Error occurred while generating COG for low-resolution image: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

import os
import base64
# Initialize Gradio Inference object
inference = GradioInference()
# Path to the local SVG file (modify this path if it's located in another folder)
local_svg_path = "assets/banner.svg"  # Update this with the actual path to your SVG file
# Function to convert the SVG file to Base64
def svg_to_base64(file_path):
    with open(file_path, "rb") as f:
        svg_base64 = base64.b64encode(f.read()).decode("utf-8")
    return svg_base64

if not os.path.exists(local_svg_path):
    raise FileNotFoundError(f"SVG file not found at {local_svg_path}")

# Convert SVG to Base64
svg_base64 = svg_to_base64(local_svg_path)
# Create the HTML to embed the SVG using the Base64 data
banner_html = f"""
<div style="text-align: center; margin-bottom: 20px;">
    <img src="data:image/svg+xml;base64,{svg_base64}" alt="Banner" style="width: 100%; max-width: 1200px;" />
</div>
"""

# Define the theme for the Gradio Blocks
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange", secondary_hue="gray")) as demo:
    # Add a banner with the embedded SVG
    gr.HTML(value=banner_html)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            plot_output = gr.Plot()
            # date_picker = Calendar(type="date", label="Select Date", info="Pick a date from the calendar.")
            date_picker = gr.DateTime(label="Select Date and Time")


            # Run Inference button will automatically take the "primary_hue" color from the theme
            run_inference = gr.Button("Run Inference", variant="primary")
            cog_button = gr.Button("Generate Super-Resolution COG file")
            lr_cog_button = gr.Button("Generate Low-Resolution COG file")
            # Add file outputs for downloading
            output_cog = gr.File(label="Download Super-Resolution COG file")
            output_lr_cog = gr.File(label="Download Low-Resolution COG file")

    # Link the input and output components to the GradioInference class
    run_inference.click(
        fn=inference.run_inference_on_date,
        inputs=date_picker,
        outputs=plot_output,
    )

    cog_button.click(
        fn=inference.generate_cog_file,
        inputs=None,
        outputs=output_cog,  # Use gr.File as output
    )

    lr_cog_button.click(
        fn=inference.generate_lr_image_cog_file,
        inputs=None,
        outputs=output_lr_cog,  # Use gr.File as output
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)