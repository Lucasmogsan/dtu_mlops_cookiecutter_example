import torch
import pytest
import os.path
from dtu_mlops_cookiecutter_example.data.make_dataset import mnist
from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT, _NON_EXISTING_PATH

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")  # skip test if data is not found
def test_data():
    N_train = 25000
    N_test = 5000

    train_data, test_data = mnist(_PATH_DATA)

    assert len(train_data) == N_train and len(test_data) == N_test, "Dataset does not have the correct number of samples"
    assert train_data[0][0].shape == torch.Size([1, 28, 28]), "Train data does not have the correct shape"


def test_error_on_wrong_shape():
    # Verifies that the function raises a ValueError if the path to data is not specified to the function
    with pytest.raises(ValueError, match='No path to data specified'):
        _, _ = mnist()



@pytest.mark.skipif(not os.path.exists(_NON_EXISTING_PATH), reason="Data files not found")
def test_something_about_data():
    N_train = 25000
    N_test = 5000
    train_data, test_data = mnist(_NON_EXISTING_PATH)
    assert len(train_data) == N_train and len(test_data) == N_test, "Dataset does not have the correct number of samples"
    assert train_data[0][0].shape == torch.Size([1, 28, 28]), "Train data does not have the correct shape"
        


if __name__ == "__main__":
    test_data()