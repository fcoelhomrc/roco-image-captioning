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
    }
}

with open("config.yaml", "w") as outfile:
    yaml.dump(config, outfile, default_flow_style=False)