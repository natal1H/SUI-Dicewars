import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import SGD
import matplotlib.pyplot as plt

# TODO: niekde bola chyba pri vytváraní datasetu, stále dáva iba rovnakého víťaza (4)
# skontrolovať tak, že spojiť všetky súbory do jedného veľkého a v ňom spočítať výskyty jednotlivých víťazov

BATCH_SIZE = 32
EPOCHS = 25
VAL_SPLIT = 0.2
DATASET_LOC = "./dataset2507/00242.npy"


def relu(x):
    return x * F.relu(x)


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(633, 256)
        self.b1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.b2 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):

        x = relu(self.fc1(x))
        x = self.b1(x)
        x = relu(self.fc2(x))
        x = self.b2(x)
        x = self.fc4(x)

        return x


class CustomDatasetFromNpy(Dataset):
    def __init__(self, npy_path):
        game_arr = torch.from_numpy(np.load(npy_path))
        self.data = game_arr[:, ..., :-1].float()  # from all rows take all elements expect last one (game states - x - data)
        self.labels = game_arr[:, ..., -1].long()  # from all rows take only last element (the winners - y - labels)

    def __getitem__(self, index):
        # This method should return only 1 sample and label
        dato = self.data[index]
        label = self.labels[index]
        return dato, label

    def __len__(self):
        return len(self.labels)


def get_train_val_loader(location, val_split=0.2, batch_size=32, shuffle=True):
    dataset = CustomDatasetFromNpy(location)
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, validation_loader


def run_gradient_descent(model, batch_size=32, learning_rate=0.01, weight_decay=0, num_epochs=25):
    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    iters, losses = [], []
    iters_sub, train_acc, val_acc = [], [], []
    train_loader, val_loader = get_train_val_loader(DATASET_LOC, VAL_SPLIT, batch_size, True)

    # training
    n = 0  # the number of iterations
    for epoch in range(num_epochs):
        for xs, ts in iter(train_loader):
            if len(ts) != batch_size:
                continue

            # xs = xs.view(-1, 784)  # flatten the image. The -1 is a wildcard # asi netreba keď to už máme 1D
            zs = model(xs)
            _loss = loss(zs, ts)  # compute the total loss
            _loss.backward()  # compute updates for each parameter
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(_loss) / batch_size)  # compute *average* loss
            if n % 10 == 0:
                iters_sub.append(n)

                dataset = CustomDatasetFromNpy(DATASET_LOC)

                # Creating data indices for training and validation splits:
                dataset_size = len(dataset)
                split = int(np.floor(VAL_SPLIT * dataset_size))
                indices = list(range(dataset_size))
                train_indices, val_indices = indices[split:], indices[:split]

                # Creating PT data samplers and loaders:
                train_sampler = SubsetRandomSampler(train_indices)
                valid_sampler = SubsetRandomSampler(val_indices)

                train = torch.utils.data.DataLoader(dataset, batch_size=len(train_indices), sampler=train_sampler)
                val = torch.utils.data.DataLoader(dataset, batch_size=len(val_indices), sampler=valid_sampler)
                train_acc.append(get_accuracy(model, train))
                val_acc.append(get_accuracy(model, val))

            # increment the iteration number
            n += 1

    # plotting
    plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, learning_rate))
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, learning_rate))
    plt.plot(iters_sub, train_acc, label="Train")
    plt.plot(iters_sub, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    return model


def get_accuracy(model, loader):
    correct, total = 0, 0
    for xs, ts in loader:
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1]  # get the index of the max logit
        correct += pred.eq(ts.view_as(pred)).sum().item()
        total += int(ts.shape[0])
        return correct / total


if __name__ == "__main__":
    model = Network()
    run_gradient_descent(model, batch_size=BATCH_SIZE, learning_rate=0.01, num_epochs=EPOCHS)
