# Streaming Deep Learning for Super Resolution on DestinE Climate Data

This repository is a demonstrator for using Streaming Deep Learning to perform Super Resolution on DestinE Climate Data. It leverages a ResNet model with 8x super-resolution capability, applied to climate datasets streamed directly from the Earth DataHub service. The repository also provides a web application for visualizing inference results and generating GeoTIFF files from both low-resolution/high-resolution outputs.

## Key Features:

$\textcolor{orange}{\textsf{Model Architecture}}$: ResNet architecture designed for an 8x super-resolution task.
Data Streaming: Climate data are streamed via the [Earth DataHub service](https://earthdatahub.destine.eu/collections/d1-climate-dt-ScenarioMIP-SSP3-7.0-IFS-NEMO/datasets/0001-high-sfc) into a DataLoader for real-time processing.

$\textcolor{orange}{\textsf{Data}}$:

- [Low Resolution (LR)](https://earthdatahub.destine.eu/collections/d1-climate-dt-ScenarioMIP-SSP3-7.0-IFS-NEMO/datasets/0001-standard-sfc): Climate Digital Twin (DT) temperature at 2 meters (t2m), IFS-NEMO model, hourly data on single levels.
- [Ground Truth (HR)](https://earthdatahub.destine.eu/collections/d1-climate-dt-ScenarioMIP-SSP3-7.0-IFS-NEMO/datasets/0001-high-sfc): High-Resolution (HR) Climate Digital Twin temperature at 2 meters (t2m), IFS-NEMO model, hourly data on single levels.

$\textcolor{orange}{\textsf{Data Streaming}}$: Climate data are streamed via the Earth DataHub service into a DataLoader for real-time processing without the need to download them locally.


$\textcolor{orange}{\textsf{Web Application}}$: To visualize inference results and generating GeoTIFF files from both low-resolution/high-resolution outputs.


## Prerequisites

1. Install Python
    Download and install Python
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh
    ```

2. Clone the repository:
    ```bash
    git git@github.com:tetaud-sebastien/DestinE_eXchange_SR.git
    cd DestinE_eXchange_SR
    ```

3. Install the required packages
    Create python environment:
    ```bash
    conda create --name env python==3.12
    ```
    Activate the environment

    ```bash
    conda activate env
    ```
    Install python package
    ```Bash
    pip install -r requirements.txt
    ```
## Train model

It is possible to train the model either on the following notebook: **notebook**

The training script takes a configuration file as input, which parses the training parameters(TODO).
You can also run the script directly using the following command:
```Bash
python train.py
```

## Test Model with Gradio

```Bash
python serve.py
```

## Future Work

- Expand to N Parameters: Although the current setup processes only one variable (t2m), the trainer will be extended to handle multiple parameters simultaneously.
- Multiple Sources: The architecture is flexible enough to incorporate data from various sources, allowing for a multi-source approach to super-resolution tasks in climate modeling

## Getting Help

Feel free to ask questions at the following email adress: [sebastien.tetaud@esa.int](sebastien.tetaud@eas.int) or open a ticket.