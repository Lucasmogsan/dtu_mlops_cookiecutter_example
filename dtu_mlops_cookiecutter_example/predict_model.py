import torch
import click
import os
import numpy as np


@click.group()
def cli():
    """Command line interface."""
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths:
current_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(current_dir, "models")

package_dir = os.path.join(current_dir, "../")
raw_data_dir = os.path.join(package_dir, "data/raw/corruptmnist")
processed_data_dir = os.path.join(package_dir, "data/processed")
visualization_dir = os.path.join(package_dir, "reports/figures")


## PREDICTION
@click.command()
@click.argument(
    "model_name"
)  # help="name of model to evaluate - remember to add .pt and should be loated in models folder"
@click.argument("input_images")  # help="path from package root to input images to predict on - remember to add .pt"
def predict(model_name: torch.nn.Module, input_images) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    print("Predicting...")

    # Import model
    model = torch.load(os.path.join(models_dir, model_name))

    # Import data
    input_path = os.path.join(package_dir, input_images)
    if input_path.endswith(".pt"):  # If input path ends with .pt, it is a tensor
        input_data = torch.load(input_path)
    elif input_path.endswith(".npy"):  # If input path ends with .npy, it is a numpy array
        input_data = torch.from_numpy(np.load(input_path)).unsqueeze(1)

    predictions = []

    # Evaluation loop
    with torch.no_grad():
        for image in input_data:
            image = image.to(device).unsqueeze(1)

            output = model(image)

            predictions.append(output.argmax(dim=1).cpu())

        predictions = torch.cat(predictions, dim=0)

    print("Predictions: ", predictions)
    return predictions


cli.add_command(predict)

if __name__ == "__main__":
    cli()
