import yaml
import torch

from data_utils import load_dataset, perform_split

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    original_data = load_dataset(cfg["dataset_name"])
    train_td, val_td, test_td = perform_split(
        original_data, "transductive", cfg["split_ratios"], cfg["seed"], cfg["task"]
    )
    train_ind, val_ind, test_ind = perform_split(
        original_data, "inductive", cfg["split_ratios"], cfg["seed"], cfg["task"]
    )

    assert torch.allclose(train_td.edge_index, train_ind.edge_index)
    assert torch.allclose(val_td.edge_label_index, val_ind.edge_label_index)
