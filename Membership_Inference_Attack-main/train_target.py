from shadow.trainer import train
from shadow.make_data import make_member_nonmember
from utils.seed import seed_everything
from utils.load_config import load_config
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from easydict import EasyDict
import yaml
import wandb
import importlib

# load config
CFG = load_config("CFG")


# seed for future replication
seed_everything(CFG.seed)

# Load the CIFAR dataset
# CIFAR train is used for SHADOW MODEL train & evaluation whereas CIFAR test is used for TARGET MODEL train & evaluation
if CFG.dataset_name.lower() == "cifar10":
    DSET_CLASS = torchvision.datasets.CIFAR10
    CFG.num_classes = 10
elif CFG.dataset_name.lower() == "cifar100":
    DSET_CLASS = torchvision.datasets.CIFAR100
    CFG.num_classes = 100

transform = transforms.Compose(
    [
        transforms.Resize((CFG.input_resolution, CFG.input_resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

testset = DSET_CLASS(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=CFG.val_batch_size, shuffle=False, num_workers=2)

# define dataset for attack model that shadow models will generate
print("mapped classes to ids:", testset.class_to_idx)

# Training multiple shadow models
model_architecture = importlib.import_module("torchvision.models")
model_class = getattr(model_architecture, CFG.model_architecture)
criterion = nn.CrossEntropyLoss()

# Train Target Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Define Devices
target_model = model_class(pretrained=CFG.bool_pretrained)
target_model.fc = nn.Linear(in_features=target_model.fc.in_features, out_features=CFG.num_classes)
target_model = target_model.to(device)
optimizer = AdamW(target_model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)

target_train_indices = np.random.choice(len(testset), CFG.target_train_size, replace=False)
target_eval_indices = np.setdiff1d(np.arange(len(testset)), target_train_indices)
# save target_train_indices as dataframe
pd.DataFrame(target_train_indices, columns=["index"]).to_csv(
    "./attack/train_indices.csv", index=False
)

subset_tgt_train = Subset(testset, target_train_indices)
subset_tgt_eval = Subset(testset, target_eval_indices)

subset_tgt_train_loader = DataLoader(
    subset_tgt_train, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2
)
subset_tgt_eval_loader = DataLoader(
    subset_tgt_eval, batch_size=CFG.val_batch_size, shuffle=False, num_workers=2
)

run_name = f"{model_architecture}_target"
wandb.init(
    entity="9-nine",
    project="membership_inference_attack",
    group=f"{target_model.__class__.__name__}_target",
    name=run_name,
)

train(
    CFG,
    target_model,
    subset_tgt_train_loader,
    subset_tgt_eval_loader,
    optimizer,
    CFG.save_path,
    shadow_number=-1,
    criterion=criterion,
    scheduler=None,
)

wandb.finish()
