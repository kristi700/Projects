import yaml
import torch

from models import EdgeGNN
from data_utils import load_dataset, perform_split


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    original_data = load_dataset(cfg["data"]["dataset_name"])
    train_td, val_td, test_td = perform_split(
        original_data, "transductive", cfg["data"]["split_ratios"], cfg["data"]["seed"], cfg["task"]
    )
    train_ind, val_ind, test_ind = perform_split(
        original_data, "inductive", cfg["data"]["split_ratios"], cfg["data"]["seed"], cfg["task"]
    )

    assert torch.allclose(train_td.edge_index, train_ind.edge_index)
    assert torch.allclose(val_td.edge_label_index, val_ind.edge_label_index)

    # For convinience
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    if train_cfg['device'] == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(train_cfg['device'])
    print(f"Using device: {DEVICE}")

    model = EdgeGNN(
        in_channels=original_data.num_node_features,
        hidden_channels=model_cfg['hidden_channels'],
        num_layers=model_cfg['num_layers'],
        dropout_rate=model_cfg['dropout_rate'],
        use_skip_connection=model_cfg['use_skip_connection'],
        use_layer_norm=model_cfg['use_layer_norm'],
        use_bias=model_cfg['use_bias'],
        eps_trainable=model_cfg['eps_trainable'],
        task_type=cfg['task'],
        use_input_norm=model_cfg['use_input_norm']
    ).to(DEVICE)

    ## test
    dummy_data = train_td.to(DEVICE)
    model.eval()
    with torch.no_grad():
        node_embeddings = model(dummy_data.x, dummy_data.edge_index)
