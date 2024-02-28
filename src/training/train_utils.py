import torch


def train(model, loader, f_loss, optimizer, device):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Returns :
    """
    model.train()
    # with tqdm.tqdm(loader, unit="batch") as tepoch:
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        if hasattr(model, "penalty"):
            model.penalty().backward()
        optimizer.step()
        # val_loss, val_acc = test(model, valid_loader, f_loss, device)
        # tepoch.set_postfix(loss = val_loss, accuracy=100. * val_acc)


