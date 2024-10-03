import gradio as gr
from gradio_calendar import Calendar
import xarray as xr
import datetime
import torch
from model import SRResNet  # Replace with actual import paths
import torch
from torchvision import transforms
from loguru import logger
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from super_resolution_inference import SuperResolutionInference


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
    # lr = lr.sel(time="2020-08")
    # Model path and SRResNet initialization
    model_path = "/home/ubuntu/project/DestinE_eXchange_SR/lightning_logs/version_10/checkpoints/best-val-ssim-epoch=91-val_ssim=0.43.pt"
    model = SRResNet(large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=8)

    # Initialize the inference class
    sr = SuperResolutionInference(model_path=model_path, model_class=model)

    formatted_str = selected_date.strftime('%Y-%m-%dT%H:%M:%S')
    # print(selected_date)
    # print(formatted_str)
    lr_image = lr.sel(time=formatted_str)
    lr_mean = 16.192955017089844
    lr_std = 4.557614326477051

    # Preprocess the image
    preprocessed_image = sr.preprocess(lr_image, lr_mean=lr_mean, lr_std=lr_std)

    # Perform inference
    sr_result = sr.inference(preprocessed_image)

    # Visualize the results
    fig = sr.visualize(lr_image=lr.sel(time=formatted_str), sr_image=sr_result, lr_time=selected_date)

    return fig


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):  # Adjust min_width to make the button and calendar the same size
            # Add the plot output first
            plot_output = gr.Plot()

            # Add your input calendar below the plot
            date_picker = Calendar(type="date", label="Select Date", info="Pick a date from the calendar.")

            # Add the button with same width as calendar
            run_button = gr.Button("Run Inference")

    # Link the input and output components to the function
    run_button.click(
        fn=run_inference_on_date,
        inputs=date_picker,
        outputs=plot_output,
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)
