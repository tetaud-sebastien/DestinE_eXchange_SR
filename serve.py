import gradio as gr
from gradio_calendar import Calendar
import xarray as xr
import numpy as np
import subprocess
import os
import datetime
import rioxarray  # Make sure to import rioxarray
from model import SRResNet  # Replace with actual import paths
from super_resolution_inference import SuperResolutionInference
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

# Define preprocessing and inference function
def run_inference_on_date(selected_date):
    """
    Run super-resolution inference for the selected date.

    Args:
        selected_date (str): The date to use for selecting the low-resolution data.

    Returns:
        tuple: (Low-resolution image, Super-resolved image)
    """

    data = xr.open_dataset(
    "https://cacheb.dcms.destine.eu/d1-climate-dt/ScenarioMIP-SSP3-7.0-IFS-NEMO-0001-standard-sfc-v0.zarr",
    engine="zarr",
    storage_options={"client_kwargs": {"trust_env": "true"}},
    chunks={},
    )
    t2m_lr = data.t2m.astype("float32") - 273.15
    t2m_lr.attrs["units"] = "C"
    lr = t2m_lr.sel(**{"latitude": slice(47, 58.22), "longitude": slice(5, 16.25)})
    model_path = "/home/ubuntu/project/DestinE_eXchange_SR/lightning_logs/version_46/checkpoints/best-val-ssim-epoch=97-val_ssim=0.59.pt"
    model = SRResNet(large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=8)
    # Initialize the inference class
    sr = SuperResolutionInference(model_path=model_path, model_class=model)
    formatted_str = selected_date.strftime('%Y-%m-%dT%H:%M:%S')
    lr_image = lr.sel(time=formatted_str)
    lr_mean = 10.007193565368652
    lr_std = 4.303609371185303
    hr_mean = 10.094209671020508
    hr_std = 4.23423957824707
    # Preprocess the image
    preprocessed_image = sr.preprocess(lr_image, lr_mean=lr_mean, lr_std=lr_std)
    # Perform inference
    sr_result = sr.inference(preprocessed_image)
    sr_result = sr.postprocessing(sr_result, hr_mean, hr_std)
    fig = sr.visualize(lr_image=lr_image, sr_image=sr_result, lr_time=lr_image.time.values)
    return fig

# Define the generate_cog_file function
def generate_cog_file():
    try:
        # Step 1: Create the xarray dataset with t2m variable
        time_series = np.random.randn(1, 4096, 8193)  # Random data for example
        time_stamps = [datetime.datetime.now()]  # Current timestamp for example
        latitudes = np.linspace(-90, 90, 4096)
        longitudes = np.linspace(-180, 180, 8193)

        ds = xr.Dataset(
            data_vars={"t2m": (["latitude", "longitude"], time_series[0])},
            coords={"latitude": latitudes, "longitude": longitudes, "time": time_stamps[0]},
            attrs={"description": "2-meter temperature"}
        )
        var = ds['t2m'] - 273.15  # Convert to Celsius
        var = var.rename({'latitude': 'y', 'longitude': 'x'})
        var.rio.write_crs("EPSG:4326", inplace=True)
        # Handle NaN values and save as a TIFF file
        var_filled = var.fillna(-9999)
        tif_filename = 'tif_filename.tif'
        var_filled.rio.to_raster(tif_filename)

        # Step 2: Run the subprocess commands for creating COG
        vrt_filename = 'vrt_filename.vrt'
        output_vrt_filename = 'output_vrt_filename.vrt'
        output_cog_filename = 'output_cog_filename.tif'
        dte_output_cog_optimised_filename = 'dte_test_cog_optimised_filename.tif'

        # Convert to VRT
        subprocess.run(f"gdal_translate -of VRT {tif_filename} {vrt_filename}", shell=True, check=True)

        # Apply color relief
        subprocess.run(f"gdaldem color-relief {vrt_filename} colormap.txt {output_vrt_filename}", shell=True, check=True)

        # Convert VRT to COG
        subprocess.run(f"gdal_translate -of COG {output_vrt_filename} {output_cog_filename}", shell=True, check=True)

        # Optimize COG
        # subprocess.run(f"rio cogeo create {output_cog_filename} {dte_output_cog_optimised_filename} --blocksize 512 --overview-resampling average --overview-level 8 --web-optimized", shell=True, check=True)

        # Return success message
        return f"COG file created successfully: {dte_output_cog_optimised_filename}"

    except subprocess.CalledProcessError as e:
        return f"Error occurred: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):  # Adjust min_width to make the button and calendar the same size
            # Add the plot output first
            plot_output = gr.Plot()

            # Add your input calendar below the plot
            date_picker = Calendar(type="date", label="Select Date", info="Pick a date from the calendar.")

            # Add the button with same width as calendar
            run_button = gr.Button("Run Inference")
            cog_button = gr.Button("Generate COG file")

            # Add text output to display terminal message
            terminal_output = gr.Textbox(label="Processing information", placeholder="...")


    # Link the input and output components to the function
    run_button.click(
        fn=run_inference_on_date,
        inputs=date_picker,
        outputs=plot_output,
    )

        # Link the cog_button to the generate_cog_file function and display the output in the Textbox
    cog_button.click(
        fn=generate_cog_file,
        inputs=None,
        outputs=terminal_output  # Display the returned message in the Textbox
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)