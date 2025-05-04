import yaml
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
import traceback
from torch_geometric.data import Data
from collections import defaultdict

from models import EdgeGNN
from data_utils import load_dataset, perform_split, seed_everything


def plot_results(results, epochs, save_dir="plots"):
    """Generates and saves plots for training history and test results."""
    print(f"\n--- Generating Plots (saving to '{save_dir}') ---")
    os.makedirs(save_dir, exist_ok=True)

    split_names = list(results.keys())
    if not split_names:
        print("No results available to plot.")
        return

    plt.figure(figsize=(10, 6))
    for name in split_names:
        if "history" in results[name] and not results[name].get("error"):
            history = results[name]["history"]
            epochs_ran = history.get(
                "epoch", list(range(1, len(history["train_loss"]) + 1))
            )
            plt.plot(
                epochs_ran,
                history["train_loss"],
                label=f"{name.capitalize()} Train Loss",
            )
            plt.plot(
                epochs_ran,
                history["val_loss"],
                label=f"{name.capitalize()} Val Loss",
                linestyle="--",
            )
        else:
            print(f"Skipping loss plot for {name} due to missing history or error.")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curves")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.close()
    print("Saved loss_curves.png")

    plot_auc = False
    plt.figure(figsize=(10, 6))
    for name in split_names:
        if (
            "history" in results[name]
            and "val_auc" in results[name]["history"]
            and not results[name].get("error")
        ):
            history = results[name]["history"]

            valid_auc_indices = [
                i for i, auc in enumerate(history["val_auc"]) if not np.isnan(auc)
            ]
            if valid_auc_indices:
                epochs_ran = [
                    history.get("epoch", list(range(1, len(history["val_auc"]) + 1)))[i]
                    for i in valid_auc_indices
                ]
                valid_auc = [history["val_auc"][i] for i in valid_auc_indices]
                plt.plot(epochs_ran, valid_auc, label=f"{name.capitalize()} Val AUC")
                plot_auc = True
            else:
                print(f"No valid Val AUC data to plot for {name}.")

    if plot_auc:
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.title("Validation AUC Curve")
        plt.ylim(bottom=max(0.0, plt.ylim()[0]))
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "validation_auc_curve.png"))
        print("Saved validation_auc_curve.png")
    else:
        print("Skipping validation AUC plot (no valid data).")
    plt.close()

    valid_test_results = {
        k: v["test"] for k, v in results.items() if "test" in v and not v.get("error")
    }
    if not valid_test_results:
        print("No valid test results available for bar chart.")
        return

    metrics_to_plot = sorted(list(next(iter(valid_test_results.values())).keys()))
    num_metrics = len(metrics_to_plot)
    num_splits = len(valid_test_results)
    bar_width = 0.35
    index = np.arange(num_metrics)

    fig, ax = plt.subplots(figsize=(max(6, num_metrics * 1.5), 6))

    bar_positions = {}
    split_list = sorted(list(valid_test_results.keys()))

    for i, name in enumerate(split_list):
        test_res = valid_test_results[name]
        values = [test_res.get(metric, 0) for metric in metrics_to_plot]
        pos = index + i * bar_width - (bar_width * (num_splits - 1) / 2)
        bar_positions[name] = pos
        rects = ax.bar(pos, values, bar_width, label=name.capitalize())

    ax.set_ylabel("Score")
    ax.set_title("Final Test Performance Comparison")
    ax.set_xticks(index)
    ax.set_xticklabels([m.capitalize() for m in metrics_to_plot])
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_results_comparison.png"))
    plt.close()
    print("Saved test_results_comparison.png")


def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Creates an optimizer based on the configuration."""
    optimizer_name = config.get("optimizer", "Adam")
    lr = config.get("learning_rate", 0.01)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    Runs the training loop, including validation.
    Returns the trained model AND a history dictionary.
    """
    print(f"\n--- Training ({split_name.capitalize()}) ---")
    history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        node_emb = model(train_ds.x, train_ds.edge_index)

        if (
            not hasattr(train_ds, "edge_label_index")
            or train_ds.edge_label_index is None
        ):
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
        val_auc = val_metrics.get("auc", float("nan"))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {train_loss.item():.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc:.4f}"
            )

    return model, history


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
        train_history = None
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

        model, train_history = run_training(
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

        results[split_name] = {"history": dict(train_history)}

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
        results[split_name]["test"] = test_results

    print("\n===== FINAL RESULTS SUMMARY TABLE =====")
    valid_test_results = {
        k: v["test"]
        for k, v in results.items()
        if "test" in v and not v.get("error")
    }

    if not valid_test_results:
        print("No valid test results found for summary table.")
    else:
        print(f"{'Metric':<15} | {'Transductive':<15} | {'Inductive':<15}")
        print("-" * 50)

        all_metrics = set()
        if "transductive" in valid_test_results:
            all_metrics.update(valid_test_results["transductive"].keys())
        if "inductive" in valid_test_results:
            all_metrics.update(valid_test_results["inductive"].keys())

        for metric in sorted(list(all_metrics)):
            td_val = (
                results.get("transductive", {})
                .get("test", {})
                .get(metric, float("nan"))
            )
            ind_val = (
                results.get("inductive", {})
                .get("test", {})
                .get(metric, float("nan"))
            )
            td_err = "error" in results.get("transductive", {})
            ind_err = "error" in results.get("inductive", {})
            td_str = "ERROR" if td_err else f"{td_val:<15.4f}"
            ind_str = "ERROR" if ind_err else f"{ind_val:<15.4f}"
            print(f"{metric.capitalize():<15} | {td_str:<15} | {ind_str:<15}")

    print("\n--- Execution Finished ---")
