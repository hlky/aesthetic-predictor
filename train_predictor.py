import argparse

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument("--x-npy", type=str, required=True, help="path to the x numpy file")
parser.add_argument("--y-npy", type=str, required=True, help="path to the y numpy file")
parser.add_argument(
    "--output-dir", type=str, default=".", help="path to the output directory"
)
parser.add_argument(
    "--save-name", type=str, default="model.pth", help="name of the saved model"
)
parser.add_argument("--batch-size", type=int, default=2048, help="batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument(
    "--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="optimizer"
)
parser.add_argument(
    "--val-percentage",
    type=float,
    default=0.05,
    help="percentage of data to use for validation",
)
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument(
    "--val-count", type=int, default=10, help="number of validation samples"
)

args = parser.parse_args()


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        return optimizer


x = np.load(args.x_npy)

y = np.load(args.y_npy)

val_percentage = args.val_percentage

train_border = int(len(x) * (1 - val_percentage))

train_tensor_x = torch.Tensor(x[:train_border])
train_tensor_y = torch.Tensor(y[:train_border])

train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


val_tensor_x = torch.Tensor(x[train_border:])
val_tensor_y = torch.Tensor(y[train_border:])

val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(1024).to(device)

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# choose the loss you want to optimze for
criterion = nn.MSELoss()
criterion2 = nn.L1Loss()
epochs = args.epochs
model.train()
best_loss = 999
save_name = args.output_dir + "/" + args.save_name

for epoch in range(epochs):
    losses = []
    losses2 = []
    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if batch_num % 1000 == 0:
            print(
                "\tEpoch %d | Batch %d | Loss %6.2f" % (epoch, batch_num, loss.item())
            )
    print("Epoch %d | Loss %6.2f" % (epoch, sum(losses) / len(losses)))
    losses = []
    losses2 = []
    for batch_num, input_data in enumerate(val_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        lossMAE = criterion2(output, y)
        losses.append(loss.item())
        losses2.append(lossMAE.item())
        if batch_num % 1000 == 0:
            print(
                "\tValidation - Epoch %d | Batch %d | MSE Loss %6.2f"
                % (epoch, batch_num, loss.item())
            )
            print(
                "\tValidation - Epoch %d | Batch %d | MAE Loss %6.2f"
                % (epoch, batch_num, lossMAE.item())
            )
            # print(y)
    print("Validation - Epoch %d | MSE Loss %6.2f" % (epoch, sum(losses) / len(losses)))
    print(
        "Validation - Epoch %d | MAE Loss %6.2f" % (epoch, sum(losses2) / len(losses2))
    )
    if sum(losses) / len(losses) < best_loss:
        print("Best MAE Val loss so far. Saving model")
        best_loss = sum(losses) / len(losses)
        print(best_loss)
        torch.save(model.state_dict(), save_name)
torch.save(model.state_dict(), save_name)
print(best_loss)
print("training done")
print("inferece test with dummy samples from the val set, sanity check")
model.eval()
output = model(x[: args.val_count].to(device))
print(output.size())
print(output)
