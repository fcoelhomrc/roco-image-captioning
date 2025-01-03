import yaml

config = {
    "dataset": {
        "train_json": "/home/felipe/Projects/roco-image-captioning/roco-dataset/json/train_dataset.json",
        "validation_json": "/home/felipe/Projects/roco-image-captioning/roco-dataset/json/validation_dataset.json",
        "test_json": "/home/felipe/Projects/roco-image-captioning/roco-dataset/json/test_dataset.json",
    },
    "input_size": [224, 224],
    "train": {
        "max_seq_length": 70,
        "max_epochs": 50,
        "batch_size": 64,
        'limit_train_batches': 1.000,
        'early_stopping_patience': 5,
        'log_every_n_steps': 4,
    },
    "optimizer": {
        'method': 'adam',
        'weight_decay': 5e-4,
        'lr': 1e-5,
    },
    "registry": {
        'project': "roco-debug",
        'name': "roco-clip",
        'wandb_root_dir': "outputs/wandb_outputs",
        'checkpoints_dir': "outputs/model_checkpoints",
    }
}

with open("config.yaml", "w") as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

from pprint import pprint
pprint(config)