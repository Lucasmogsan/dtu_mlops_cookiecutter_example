import click
import torch
from models.model import myawesomemodel

import os
import matplotlib.pyplot as plt


@click.group()
def cli():
    """Command line interface."""
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths:
current_dir = os.path.dirname(os.path.realpath(__file__))
raw_data_dir = os.path.join(current_dir, '../data/raw/corruptmnist')
processed_data_dir = os.path.join(current_dir, '../data/processed')
models_dir = os.path.join(current_dir, 'models')
visualization_dir = os.path.join(current_dir, '../reports/figures')



## TRAINING
@click.command()
# Adding command line options
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
@click.option("--batch_size", default=256, help="batch size to use for training")
# Defining the function
def train(lr, epochs, batch_size):
    """Train a model on MNIST."""
    print("Training day and night")

    # Import model
    model = myawesomemodel.to(device)

    # Import data
    test_data = torch.load(processed_data_dir + '/test_images.pt')
    test_labels = torch.load(processed_data_dir + '/test_target.pt')
    # Convert to dataloader (to convert to batches)
    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    # Train model hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = epochs

    # For visualization
    train_loss = []

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)

            loss = criterion(output, labels)
            
            loss.backward()

            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        train_loss.append(loss.item())
    
    # Make plot and save as png
    plt.plot(train_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.savefig(visualization_dir + '/train_loss.png')
    

    # Save model
    return torch.save(model, models_dir + "/model.pt")


        


cli.add_command(train)


if __name__ == "__main__":
    cli()
