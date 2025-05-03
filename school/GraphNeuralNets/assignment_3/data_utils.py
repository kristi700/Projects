import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import is_undirected, to_undirected
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
        data (Data): The PyG Data object.
        strategy (str): The splitting strategy ('inductive' or 'transductive').
        ratios (dict): Dictionary defining split ratios.
        seed (int): The random seed for reproducibility.
        task_type (str): The type of edge task. Currently primarily supports
        'link_prediction' style splits via RandomLinkSplit.

    Returns:
        tuple[Data, Data, Data]: A tuple containing train_data, val_data, test_data.
    """
    print(f"\nPerforming '{strategy}' split with seed {seed}...")

    if task_type != "link_prediction":
        raise NotImplementedError(
            f"Splitting for task_type '{task_type}' "
            "not fully implemented with RandomLinkSplit focus. "
            "Need custom logic if not link prediction."
        )

    print(f"INFO: Using RandomLinkSplit with num_val={ratios['val']:.2f}, num_test={ratios['test']:.2f}.")
    seed_everything(seed)

    transform = T.RandomLinkSplit(
        num_val=ratios['val'],
        num_test=ratios['test'],
        is_undirected=True,
        add_negative_train_samples=False,
        split_labels=False,
    )

    try:
        train_data, val_data, test_data = transform(data)
    except Exception as e:
        print(f"Error during RandomLinkSplit: {e}")
        print(
            "Check if the dataset graph structure and specified ratios are compatible."
        )
        raise

    train_data.split_strategy = strategy
    val_data.split_strategy = strategy
    test_data.split_strategy = strategy

    return train_data, val_data, test_data
