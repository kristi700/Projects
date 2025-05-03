Task Overview:

    Dataset Selection:
        Choose a dataset for edge-level classification or regression from PyG’s datasets.
        Justify your choice, highlighting its relevance and challenges.

    Dataset Splitting:
        Apply different splitting strategies using fixed random seeds for reproducibility.
        Use the following edge-splitting strategies:
            Inductive link prediction split (unseen edges during training) (pick the correct data for such case)
            Transductive link prediction split (edges within a fixed graph) (pick the correct data for such case)
        Employ an 80%/20%/20% split for training, validation, and testing.
        Compare the impact of these methods on performance.

    GNN Implementation:
        Implement a customized GNN using PyG’s MessagePassing module.
        Incorporate:
            The Graph Isomorphism Network (GIN) layer with skip connections and layer normalization.
            An MLP-based message-passing mechanis for message passing.
        Add optional enhancements (e.g., input normalization, bias terms, dropout).

    Edge-Level Tasks:
        Conduct tasks like edge weight prediction, link prediction, and edge classification.
        Analyze performance across tasks and splitting strategies.

    Requirements:
        Use fixed random seeds and ensure consistent results.
        Provide a detailed analysis of:
            The inductive vs. transductive split impact.
            GIN’s effectiveness for edge-level tasks.
