import yaml
import torch
import numpy as np
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
import traceback
from torch_geometric.data import Data

from models import EdgeGNN
from data_utils import load_dataset, perform_split, seed_everything


def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Creates an optimizer based on the configuration."""
    optimizer_name = config.get("optimizer", "Adam")
    lr = config.get("learning_rate", 0.01)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_criterion(task_type: str):
    """Gets the loss function based on the task type."""

    if task_type == "link_prediction":
        return torch.nn.BCEWithLogitsLoss()
    elif task_type == "edge_classification":
        return torch.nn.CrossEntropyLoss()
    elif task_type == "edge_regression":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported task type for criterion: {task_type}")


def evaluate(
    split_name: str,
    model: EdgeGNN,
    criterion,
    train_ds: Data,
    eval_ds: Data,
    original_data: Data,
    loader_config: dict,
    device: torch.device,
    task_type: str,
):
    """
    Evaluates the model on validation or test data, handling both strategies.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        if split_name == "transductive":
            node_emb_eval = model(train_ds.x, train_ds.edge_index)
            preds = model.predict_edges(node_emb_eval, eval_ds.edge_label_index)
            labels = eval_ds.edge_label.float()
            loss = criterion(preds, labels)
            total_loss = loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        elif split_name == "inductive":
            eval_loader = LinkNeighborLoader(
                data=original_data,
                num_neighbors=loader_config.get("num_neighbors", [10, 10]),
                batch_size=loader_config.get("batch_size", 512),
                edge_label_index=eval_ds.edge_label_index,
                edge_label=eval_ds.edge_label,
                shuffle=False,
                neg_sampling_ratio=loader_config.get("neg_sampling_ratio", 1.0),
            )
            print(
                f"  (Evaluating inductive with LinkNeighborLoader, batches: {len(eval_loader)})"
            )

            processed_batches = 0
            for batch in eval_loader:
                batch = batch.to(device)

                node_emb_batch = model(batch.x, batch.edge_index)
                preds = model.predict_edges(node_emb_batch, batch.edge_label_index)
                labels = batch.edge_label.float()
                loss = criterion(preds, labels)

                total_loss += loss.item() * preds.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                processed_batches += 1

            if processed_batches == 0:
                print(
                    "Warning: Inductive eval loader produced 0 batches. Check eval_ds.edge_label_index."
                )
                return {"loss": float("nan"), "auc": 0.5}

            total_loss /= sum(p.size(0) for p in all_labels)

        else:
            raise ValueError(f"Unknown split_name in evaluate: {split_name}")

    metrics = {"loss": total_loss}
    if not all_labels or all_labels[0].numel() == 0:
        print("Warning: No labels found for metric calculation.")
        if task_type == "link_prediction":
            metrics["auc"] = 0.5
        return metrics

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if task_type == "link_prediction":
        if all_labels.numel() > 0:
            try:
                probs = torch.sigmoid(all_preds).numpy()
                labels_np = all_labels.numpy()

                if len(np.unique(labels_np)) > 1:
                    metrics["auc"] = roc_auc_score(labels_np, probs)
                else:
                    print(
                        f"Warning: Only one class ({np.unique(labels_np)[0]}) present in eval labels. AUC set to 0.5."
                    )
                    metrics["auc"] = 0.5
            except ValueError as e:
                print(f"Warning during AUC calculation: {e}")
                metrics["auc"] = 0.5
        else:
            metrics["auc"] = 0.5

    return metrics


def run_training(
    model,
    optimizer,
    criterion,
    train_ds,
    val_ds,
    original_data,
    loader_config,
    epochs,
    split_name,
    device,
    task_type,
):
    """
    Runs the training loop, including validation, adapting to split strategy.
    """
    print(f"\n--- Training ({split_name.capitalize()}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        node_emb = model(train_ds.x, train_ds.edge_index)

        if (
            not hasattr(train_ds, "edge_label_index")
            or train_ds.edge_label_index is None
        ):
            print(
                "Warning: train_ds.edge_label_index not found. Using train_ds.edge_index for positive supervision."
            )
            train_pred_idx = train_ds.edge_index
            train_target = torch.ones(train_pred_idx.size(1), device=device)
        else:
            train_pred_idx = train_ds.edge_label_index
            train_target = train_ds.edge_label.float()

        train_pred = model.predict_edges(node_emb, train_pred_idx)
        train_loss = criterion(train_pred, train_target)
        train_loss.backward()
        optimizer.step()

        val_metrics = evaluate(
            split_name,
            model,
            criterion,
            train_ds,
            val_ds,
            original_data,
            loader_config,
            device,
            task_type,
        )
        val_loss = val_metrics["loss"]
        val_auc = val_metrics.get("auc", 0.0)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {train_loss.item():.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc:.4f}"
            )

    return model


def run_test(
    model,
    criterion,
    train_ds,
    test_ds,
    original_data,
    loader_config,
    split_name,
    device,
    task_type,
):
    """Runs the testing phase using the evaluate function."""
    print(f"\n--- Testing ({split_name.capitalize()}) ---")
    test_metrics = evaluate(
        split_name,
        model,
        criterion,
        train_ds,
        test_ds,
        original_data,
        loader_config,
        device,
        task_type,
    )

    print(f"Test Results ({split_name.capitalize()}): ", end="")
    for name, value in test_metrics.items():
        print(f"{name.capitalize()}: {value:.4f} ", end="")
    print()
    return test_metrics


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    loader_cfg = cfg.get("loader", {"batch_size": 512, "num_neighbors": [10, 5]})

    if train_cfg["device"] == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(train_cfg["device"])

    print(f"--- Setup ---")
    print(f"Task Type: {cfg['task']}")

    seed = data_cfg.get("seed", 42)
    seed_everything(seed)

    try:
        original_data = load_dataset(data_cfg["dataset_name"])
        actual_in_channels = original_data.num_node_features
        print(
            f"Dataset '{data_cfg['dataset_name']}' loaded. Input features: {actual_in_channels}"
        )

        original_data = original_data.to(DEVICE)
        print("Original data moved to device (for potential inductive loader use).")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        exit()

    results = {}
    split_configs = {
        "transductive": {
            "strategy": "transductive",
            "ratios": data_cfg.get(
                "split_ratios_transductive", data_cfg["split_ratios"]
            ),
        },
        "inductive": {
            "strategy": "inductive",
            "ratios": data_cfg.get("split_ratios_inductive", data_cfg["split_ratios"]),
        },
    }

    for split_name, config in split_configs.items():
        print(f"\n===== Experiment: {split_name.upper()} =====")

        print(f"Performing {split_name} split...")
        try:
            data_to_split = original_data.clone().cpu()
            train_ds, val_ds, test_ds = perform_split(
                data_to_split,
                strategy=config["strategy"],
                ratios=config["ratios"],
                seed=seed,
                task_type=cfg["task"],
            )

            train_ds.to(DEVICE)
            val_ds.to(DEVICE)
            test_ds.to(DEVICE)
            print("Split complete and relevant data moved to device.")
        except Exception as e:
            print(f"ERROR during {split_name} split: {e}")
            traceback.print_exc()
            continue

        print("Initializing model, optimizer, criterion...")
        seed_everything(seed)
        model = EdgeGNN(
            in_channels=actual_in_channels,
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

        model = run_training(
            model,
            optimizer,
            criterion,
            train_ds,
            val_ds,
            original_data,
            loader_cfg,
            epochs,
            split_name,
            DEVICE,
            cfg["task"],
        )

        test_results = run_test(
            model,
            criterion,
            train_ds,
            test_ds,
            original_data,
            loader_cfg,
            split_name,
            DEVICE,
            cfg["task"],
        )
        results[split_name] = test_results

    print("\n===== FINAL RESULTS SUMMARY =====")
    if "transductive" in results and "inductive" in results:
        valid_results = {k: v for k, v in results.items() if "error" not in v}

        if not valid_results:
            print("No valid results found for comparison.")
        else:
            print(f"{'Metric':<15} | {'Transductive':<15} | {'Inductive':<15}")
            print("-" * 50)

            ref_metrics = next(iter(valid_results.values())).keys()
            all_metrics = set(ref_metrics)

            if "transductive" in valid_results and "inductive" in valid_results:
                all_metrics |= set(valid_results["transductive"].keys()) | set(
                    valid_results["inductive"].keys()
                )

            for metric in sorted(list(all_metrics)):
                td_val = results.get("transductive", {}).get(metric, float("nan"))
                ind_val = results.get("inductive", {}).get(metric, float("nan"))
                td_err = "error" in results.get("transductive", {})
                ind_err = "error" in results.get("inductive", {})

                td_str = "ERROR" if td_err else f"{td_val:<15.4f}"
                ind_str = "ERROR" if ind_err else f"{ind_val:<15.4f}"

                print(f"{metric.capitalize():<15} | {td_str:<15} | {ind_str:<15}")
    else:
        print("Could not find results for both splits to compare.")
        print("Results:", results)

    print("\n--- Execution Finished ---")
