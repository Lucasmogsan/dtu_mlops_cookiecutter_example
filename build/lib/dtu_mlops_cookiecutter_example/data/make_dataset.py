import torch
import os



def mnist(path_data=None):
    """Return train and test dataset for MNIST."""
    print('Load MNIST data')

    if path_data is None:
        raise ValueError('No path to data specified')

    # Define paths:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    raw_dir = os.path.join(path_data, 'raw/corruptmnist')
    processed_dir = os.path.join(path_data, 'processed')

    # Load data from files:
    train_data, train_labels = [], []
    for i in range(5):
        train_data.append(torch.load(raw_dir + f'/train_images_{i}.pt'))
        train_labels.append(torch.load(raw_dir + f'/train_target_{i}.pt'))

    test_data = torch.load(raw_dir + '/test_images.pt')
    test_labels = torch.load(raw_dir + '/test_target.pt')

    # Modify data to fit the model:
    train_data = torch.cat(train_data, dim=0)   # concatenate all tensors as it consists of 5 tensors
    train_data = train_data.unsqueeze(1) # add channel dimension
    train_labels = torch.cat(train_labels, dim=0)
 
    test_data = test_data.unsqueeze(1) # add channel dimension



    return (torch.utils.data.TensorDataset(train_data, train_labels),
            torch.utils.data.TensorDataset(test_data, test_labels)
            )

def normalize():
    """
    Take the raw data in a data/raw folder,
    process them into a single tensor,
    normalize the tensor (mean 0 and std 1),
    save this intermediate representation to the data/processed folder
    """
    print('Normalize data and save to processed folder')

    # Define paths:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    raw_dir = 'data/raw/corruptmnist'
    processed_dir = 'data/processed'

    # Load raw data:
    train_data, train_labels = [], []
    for i in range(5):
        train_data.append(torch.load(raw_dir + f'/train_images_{i}.pt'))
        train_labels.append(torch.load(raw_dir + f'/train_target_{i}.pt'))

    test_data = torch.load(raw_dir + '/test_images.pt')
    test_labels = torch.load(raw_dir + '/test_target.pt')

    # Process into single tensor (only for train data):
    train_data = torch.cat(train_data, dim=0)   # concatenate all tensors as it consists of 5 tensors
    train_labels = torch.cat(train_labels, dim=0)

    # Normalize data:
    train_data = torch.nn.functional.normalize(train_data, dim=1)
    test_data = torch.nn.functional.normalize(test_data, dim=1)

    # Add channel dimension:
    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    # Save processed data:
    torch.save(train_data, processed_dir + '/train_images.pt')
    torch.save(train_labels, processed_dir + '/train_target.pt')
    torch.save(test_data, processed_dir + '/test_images.pt')
    torch.save(test_labels, processed_dir + '/test_target.pt')

    # Print to verify:
    print('Train shape: ', train_data.shape)
    print('Test shape: ', test_data.shape)
    print('Train example, mean: ', torch.mean(train_data[0]))
    print('Test example, mean: ', torch.mean(test_data[0]))

    





if __name__ == "__main__":
    print("Test of data loading")

    normalize()

    path_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    # Load data
    train_data, test_data = mnist(path_data)

    # Convert to dataloader (to convert to batches)
    train_dataloader = torch.utils.data.DataLoader( # Convert to dataloader (to convert to batches)
    train_data,
    batch_size=64,
    shuffle=True,
    )
    # Print example batch
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)
    print(images.shape)
    print(labels.shape)