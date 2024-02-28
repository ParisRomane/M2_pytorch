#### A Fully connected 2 hidden layers classifier ####

""" IMPORTS """
import torch
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import sys
from torch.utils.tensorboard import (
    SummaryWriter,
)  # dashboard local, dashboard en ligne permet de travailler à plusieurs

### LIBRAIRIE POUR EVITER DE LES RECOPIER : DEEPCS. github jeremyfix +pytorch_template_code (auto completion)
from utils.NN_utils import test, train
from models.NN_models import FullyConnected
from utils.data_utils import load_dataset_FashionMNIST_with_standardization


""" DATASET """
train_loader, valid_loader, test_loader = (
    load_dataset_FashionMNIST_with_standardization()
)

print(
    "The train set contains {} images, in {} batches".format(
        len(train_loader.dataset), len(train_loader)
    )
)
print(
    "The validation set contains {} images, in {} batches".format(
        len(valid_loader.dataset), len(valid_loader)
    )
)
print(
    "The test set contains {} images, in {} batches".format(
        len(test_loader.dataset), len(test_loader)
    )
)

""" MODEL """

# GPU usage pour calcul modèle.
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# intantiation modèle
model = FullyConnected(1 * 28 * 28, 10)
model.to(device)
""" TRAIN TEST LOOP FUNCTIONS """


""" SAUVEGARDE DES DONNÉES (TENSORBOARD, SAUVEGARDE DE MODÈLE, SUMMARY)"""


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


###################################################
# Example usage :
# 1- create the directory "./logs" if it does not exist
top_logdir = "./logs_FC"
if not os.path.exists(top_logdir):
    os.mkdir(top_logdir)

logdir = generate_unique_logpath(top_logdir, "FC")
print("Logging to {}".format(logdir))
# -> Prints out     Logging to   ./logs/FC_1
if not os.path.exists(logdir):
    os.mkdir(logdir)


class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.filepath)
            # torch.save(self.model, self.filepath)
            self.min_loss = loss


""" LOOP PRINCIPALE """

f_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 10

tensorboard_writer = SummaryWriter(log_dir=logdir)
## ouvrir un modèle sauvegarder avec torch.save
model_path = "logs/FC_1/best_model.pt"
if os.path.isfile(model_path):
    print(model.load_state_dict(torch.load(model_path)))
# Switch to eval mode sinon les poids ne s'accualisent pas
model.eval()

model_checkpoint = ModelCheckpoint(logdir + "/best_model.pt", model)
for epoch in range(epochs):
    print("Epoch {}".format(epoch))
    train(model, train_loader, f_loss, optimizer, device)
    train_loss, train_acc = test(model, train_loader, f_loss, device)
    val_loss, val_acc = test(model, valid_loader, f_loss, device)
    print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))
    model_checkpoint.update(val_loss)
    tensorboard_writer.add_scalar("metrics/train_loss", train_loss, epoch)
    tensorboard_writer.add_scalar("metrics/train_acc", train_acc, epoch)
    tensorboard_writer.add_scalar("metrics/val_loss", val_loss, epoch)
    tensorboard_writer.add_scalar("metrics/val_acc", val_acc, epoch)

summary_file = open(logdir + "/summary.txt", "w")
summary_text = """

Executed command
================
{}

Dataset
=======
FashionMNIST

Model summary
=============
{}

{} trainable parameters

Optimizer
========
{}

""".format(
    " ".join(sys.argv),
    model,
    sum(p.numel() for p in model.parameters() if p.requires_grad),
    optimizer,
)
summary_file.write(summary_text)
summary_file.close()


tensorboard_writer.add_text("Experiment summary", summary_text)
