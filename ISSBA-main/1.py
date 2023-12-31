import wandb
import os

config = dict(
    learning_rate=0.01,
    momentum=0.2,
    architecture="CNN",
    dataset_id="peds-0192",
    infra="AWS",
)

wandb.init(
    project="detect-pedestrians",
    notes="tweak baseline",
    tags=["baseline", "paper1"],
    config=config,
)