import yaml
import torch

from models import EdgeGNN
from data_utils import load_dataset, perform_split


def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Creates an optimizer based on the configuration."""
    optimizer_name = config.get("optimizer", "Adam")
    lr = config.get("learning_rate", 0.01)
    weight_decay = config.get("weight_decay", 0.0)

    print(
        f"Creating optimizer: {optimizer_name} (lr={lr}, weight_decay={weight_decay})"
    )
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_criterion(task_type: str):
    """Gets the loss function based on the task type."""
    print(f"Getting criterion for task: {task_type}")
    if task_type == "link_prediction":
        return torch.nn.BCEWithLogitsLoss()
    elif task_type == "edge_classification":
        return torch.nn.CrossEntropyLoss()
    elif task_type == "edge_regression":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported task type for criterion: {task_type}")

def run_training(model, optimizer, criterion, train_ds, val_ds, epochs, split_name):
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        node_emb = model(train_ds.x, train_ds.edge_index)
        train_pred = model.predict_edges(node_emb, train_ds.edge_label_index)
        train_target = train_ds.edge_label.float()
        train_loss = criterion(train_pred, train_target)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_emb = model(val_ds.x, val_ds.edge_index)
            val_pred = model.predict_edges(val_emb, val_ds.edge_label_index)
            val_target = val_ds.edge_label.float()
            val_loss = criterion(val_pred, val_target)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{split_name.capitalize()} Epoch {epoch:03d}/{epochs:03d} | "
                  f"Train Loss: {train_loss.item():.8f} | "
                  f"Val Loss: {val_loss.item():.8f}")
    return model


def run_test(model, criterion, test_ds, split_name):
    model.eval()
    with torch.no_grad():
        test_emb = model(test_ds.x, test_ds.edge_index)
        test_pred = model.predict_edges(test_emb, test_ds.edge_label_index)
        test_target = test_ds.edge_label.float()
        test_loss = criterion(test_pred, test_target)
    print(f"{split_name.capitalize()} Test Loss: {test_loss.item():.8f}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    if train_cfg["device"] == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(train_cfg["device"])

    print(f"Using device: {DEVICE}")
    original_data = load_dataset(cfg["data"]["dataset_name"])
    train_td, val_td, test_td = perform_split(
        original_data,
        "transductive",
        cfg["data"]["split_ratios"],
        cfg["data"]["seed"],
        cfg["task"],
    )

    train_ind, val_ind, test_ind = perform_split(
        original_data,
        "inductive",
        cfg["data"]["split_ratios"],
        cfg["data"]["seed"],
        cfg["task"],
    )

    for ds in (train_td, val_td, test_td, train_ind, val_ind, test_ind):
        ds.to(DEVICE)

    assert torch.allclose(train_td.edge_index, train_ind.edge_index)
    assert torch.allclose(val_td.edge_label_index, val_ind.edge_label_index)

    model = EdgeGNN(
        in_channels=original_data.num_node_features,
        hidden_channels=model_cfg["hidden_channels"],
        num_layers=model_cfg["num_layers"],
        dropout_rate=model_cfg["dropout_rate"],
        use_skip_connection=model_cfg["use_skip_connection"],
        use_layer_norm=model_cfg["use_layer_norm"],
        use_bias=model_cfg["use_bias"],
        eps_trainable=model_cfg["eps_trainable"],
        task_type=cfg["task"],
        use_input_norm=model_cfg["use_input_norm"],
    ).to(DEVICE)

    optimizer = get_optimizer(model, train_cfg)
    criterion = get_criterion(cfg["task"])
    epochs = train_cfg.get("epochs", 100)

    print("\n=== Transductive Split ===")
    model = run_training(model, optimizer, criterion, train_td, val_td, epochs, "transductive")
    run_test(model, criterion, test_td, "transductive")

    model = EdgeGNN(
        in_channels=original_data.num_node_features,
        hidden_channels=model_cfg["hidden_channels"],
        num_layers=model_cfg["num_layers"],
        dropout_rate=model_cfg["dropout_rate"],
        use_skip_connection=model_cfg["use_skip_connection"],
        use_layer_norm=model_cfg["use_layer_norm"],
        use_bias=model_cfg["use_bias"],
        eps_trainable=model_cfg["eps_trainable"],
        task_type=cfg["task"],
        use_input_norm=model_cfg["use_input_norm"],
    ).to(DEVICE)
    optimizer = get_optimizer(model, train_cfg)

    print("\n=== Inductive Split ===")
    model = run_training(model, optimizer, criterion, train_ind, val_ind, epochs, "inductive")
    run_test(model, criterion, test_ind, "inductive")
