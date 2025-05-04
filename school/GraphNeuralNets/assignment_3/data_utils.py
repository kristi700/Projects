import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import is_undirected, to_undirected, subgraph
from torch_geometric.data import Data
from torch_geometric import seed_everything

SUPPORTED_DATASETS = ["Cora", "CiteSeer", "PubMed"]


def load_dataset(name: str, path: str = "./data") -> tuple[Data, str]:
    """
    Loads a PyG dataset and provides justification.

    Args:
        name (str): The name of the dataset to load (e.g., 'Cora').
        path (str): The path to store/load the dataset.
    Returns:
        tuple[Data, str]: A tuple containing:
            - data (Data): The PyG Data object (usually the first graph).
            - justification (str): A textual justification for choosing this dataset.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{name}' not supported or recognized. "
            f"Supported datasets: {SUPPORTED_DATASETS}"
        )

    print(f"Loading dataset: {name}...")
    try:
        dataset = Planetoid(root=path, name=name)
        data = dataset[0]
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{name}'. Error: {e}")

    if not is_undirected(data.edge_index, num_nodes=data.num_nodes):
        print("Graph is directed. Converting to undirected.")
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    num_isolated_nodes = data.num_nodes - torch.unique(data.edge_index.flatten()).size(
        0
    )
    if num_isolated_nodes > 0:
        print(f"Warning: Dataset contains {num_isolated_nodes} isolated nodes.")

    return data


def perform_split(
    data: Data,
    strategy: str,
    ratios: dict,
    seed: int,
    task_type: str = "link_prediction",
) -> tuple[Data, Data, Data]:
    """
    Applies link prediction splitting strategies to the data.

    Args:
        data (Data): The PyG Data object (assumed to have undirected edge_index).
        strategy (str): The splitting strategy:
                        'transductive': Uses RandomLinkSplit (holds out edges).
                        'inductive': Splits nodes first, creates subgraph for training.
        ratios (dict): Dictionary defining split ratios.
                       For 'transductive', interpreted by RandomLinkSplit (e.g., val=0.1, test=0.1).
                       For 'inductive', interpreted as node ratios (e.g., train=0.8, val=0.1, test=0.1).
        seed (int): The random seed for reproducibility.
        task_type (str): The type of edge task. Inductive split here focuses on link prediction setup.

    Returns:
        tuple[Data, Data, Data]: A tuple containing train_data, val_data, test_data.
                                 Structure differs significantly between strategies.
    """
    print(f"\nPerforming '{strategy}' split with seed {seed}...")
    seed_everything(seed)

    if task_type != "link_prediction":
        raise NotImplementedError(
            "Current inductive split logic is tailored for link prediction."
        )

    if strategy == "transductive":
        print("Applying Transductive Split (RandomLinkSplit on edges)...")

        val_ratio = ratios.get("val", 0.1)
        test_ratio = ratios.get("test", 0.1)
        print(
            f"INFO: Using RandomLinkSplit with num_val={val_ratio:.2f}, num_test={test_ratio:.2f}."
        )

        transform = T.RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            is_undirected=True,
            add_negative_train_samples=False,
            split_labels=False,
        )
        try:
            train_data, val_data, test_data = transform(data)

            train_data.split_strategy = strategy
            val_data.split_strategy = strategy
            test_data.split_strategy = strategy

            print("Transductive Split details:")
            print(
                f"  Train: Message Edges={train_data.edge_index.size(1)}, Supervision Edges={train_data.edge_label_index.size(1)}"
            )
            print(f"  Val:   Supervision Edges={val_data.edge_label_index.size(1)}")
            print(f"  Test:  Supervision Edges={test_data.edge_label_index.size(1)}")
            return train_data, val_data, test_data

        except Exception as e:
            print(f"Error during Transductive RandomLinkSplit: {e}")
            raise

    elif strategy == "inductive":
        print("Applying Inductive Split (Splitting Nodes)...")
        num_nodes = data.num_nodes
        node_indices = torch.randperm(num_nodes)

        train_ratio = ratios.get("train", 0.8)
        val_ratio = ratios.get("val", 0.1)

        num_train = int(num_nodes * train_ratio)
        num_val = int(num_nodes * val_ratio)
        num_test = num_nodes - num_train - num_val

        if num_test <= 0:
            raise ValueError(
                f"Calculated non-positive number of test nodes ({num_test}). Check ratios."
            )

        train_idx = node_indices[:num_train]
        val_idx = node_indices[num_train : num_train + num_val]
        test_idx = node_indices[num_train + num_val :]

        print(
            f"Node split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
        )

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        train_edge_index, _ = subgraph(
            train_idx, data.edge_index, relabel_nodes=False, num_nodes=num_nodes
        )

        train_data = Data(
            x=data.x,
            edge_index=train_edge_index,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        train_data.split_strategy = strategy
        print(
            f"  Train Graph: Nodes={train_data.train_mask.sum()}, Edges={train_data.edge_index.size(1)}"
        )

        val_edges_mask = (
            (train_mask[data.edge_index[0]] & val_mask[data.edge_index[1]])
            | (val_mask[data.edge_index[0]] & train_mask[data.edge_index[1]])
            | (val_mask[data.edge_index[0]] & val_mask[data.edge_index[1]])
        )
        val_pos_edge_index = data.edge_index[:, val_edges_mask]
        val_pos_edge_label = torch.ones(val_pos_edge_index.size(1))

        val_data = Data(
            x=data.x,
            edge_label_index=val_pos_edge_index,
            edge_label=val_pos_edge_label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        val_data.split_strategy = strategy
        print(f"  Val Supervision: Positive Edges={val_data.edge_label_index.size(1)}")

        test_edges_mask = (
            (train_mask[data.edge_index[0]] & test_mask[data.edge_index[1]])
            | (test_mask[data.edge_index[0]] & train_mask[data.edge_index[1]])
            | (val_mask[data.edge_index[0]] & test_mask[data.edge_index[1]])
            | (test_mask[data.edge_index[0]] & val_mask[data.edge_index[1]])
            | (test_mask[data.edge_index[0]] & test_mask[data.edge_index[1]])
        )
        test_pos_edge_index = data.edge_index[:, test_edges_mask]
        test_pos_edge_label = torch.ones(test_pos_edge_index.size(1))

        test_data = Data(
            x=data.x,
            edge_label_index=test_pos_edge_index,
            edge_label=test_pos_edge_label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        test_data.split_strategy = strategy

        return train_data, val_data, test_data

    else:
        raise ValueError(
            f"Unsupported split strategy: '{strategy}'. Choose 'transductive' or 'inductive'."
        )
