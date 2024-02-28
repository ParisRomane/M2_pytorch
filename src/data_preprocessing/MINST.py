from src.data_preprocessing.data_utils import compute_mean_std,DatasetTransformer
import os
import torch
import torchvision.transforms as transforms
import torchvision

def load_dataset_FashionMNIST_with_standardization(augmentation=True):
    dataset_dir = os.path.join(os.path.expanduser("~"), "Datasets", "FashionMNIST")
    valid_ratio = 0.2  # Going to use 80%/20% split for train/valid

    # Load the dataset for the training/validation sets
    train_valid_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_dir,
        train=True,
        transform=None,  # transforms.ToTensor(),
        download=True,
    )

    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_dir, transform=None, train=False  # transforms.ToTensor(),
    )

    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid = int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(
        train_valid_dataset, [nb_train, nb_valid]
    )
    train_augment_transforms = transforms.Compose(
        [
            #transforms.ToPILImage(), #car sinon pas PIL format et donc compute_mean ne marche pas.
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]
    )

    normalizing_dataset = train_dataset
    #train_dataset = DatasetTransformer(train_dataset, train_augment_transforms)
    train_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
    valid_dataset = DatasetTransformer(valid_dataset, transforms.ToTensor())
    test_dataset = DatasetTransformer(test_dataset, transforms.ToTensor())

    ## NORMALISATION
    num_threads = 4  # Loading the dataset is using 4 CPU threads
    batch_size = 128  # Using minibatches of 128 samples si le dernier mini_batch <128, c'est pas grave, ca fait un mini_batch petit. Faire attention dans le training ( voir % accuracy)

    normalizing_dataset = train_dataset
    normalizing_loader = torch.utils.data.DataLoader(
        dataset=normalizing_dataset, batch_size=batch_size, num_workers=num_threads
    )

    # Compute mean and variance from the training set
    mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)

    data_transforms = transforms.Compose(
        [transforms.Lambda(lambda x: (x - mean_train_tensor) / std_train_tensor)]
    )

    train_dataset = DatasetTransformer(train_dataset, data_transforms)
    valid_dataset = DatasetTransformer(valid_dataset, data_transforms)
    test_dataset = DatasetTransformer(test_dataset, data_transforms)

    ## DATALOADERS

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # <-- this reshuffles the data at every epoch
        num_workers=num_threads,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
    )

    return train_loader, valid_loader, test_loader
