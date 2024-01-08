from dtu_mlops_cookiecutter_example.data.make_dataset import mnist
from test import _PATH_DATA
import torch


def test_data():
    N_train = 25000
    N_test = 5000

    train_data, test_data = mnist(_PATH_DATA)

    assert len(train_data) == N_train and len(test_data) == N_test
    assert train_data[0][0].shape == torch.Size([1, 28, 28])
    # assert that all labels are represented
    print(train_data[0][1])
    print(train_data[1][0])
    assert set([train_data[i][1] for i in range(len(train_data))]) == set(range(10))


if __name__ == "__main__":
    test_data()