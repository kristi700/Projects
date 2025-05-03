import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch import Tensor


class CustomGINConv(MessagePassing):
    """
    Custom Graph Isomorphism Network (GIN) layer based on MessagePassing.
    Includes an MLP for message transformation, optional Layer Normalization,
    and the GIN epsilon parameter.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        mlp_hidden_channels (int): Hidden dimension size for the internal MLP.
        eps (float, optional): Initial value for epsilon. Defaults to 0.0.
        train_eps (bool, optional): If True, epsilon is a learnable parameter.
                                    Defaults to True.
        use_layer_norm (bool, optional): If True, applies Layer Normalization
                                         to the output. Defaults to True.
        use_bias (bool, optional): If True, adds bias terms to the MLP layers.
                                   Defaults to True.
        **kwargs: Additional arguments for MessagePassing (e.g., aggr).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_hidden_channels: int,
        eps: float = 0.0,
        train_eps: bool = True,
        use_layer_norm: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_layer_norm = use_layer_norm

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_channels, bias=use_bias),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, out_channels, bias=use_bias),
        )

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

        if use_layer_norm:
            self.norm = nn.LayerNorm(out_channels, elementwise_affine=True)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """
        Forward pass of the GIN layer.

        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index (Adj): Graph connectivity in COO format with shape [2, num_edges].

        Returns:
            Tensor: Updated node feature matrix with shape [num_nodes, out_channels].
        """

        x_original = x

        aggregated_neighbors = self.propagate(edge_index, x=x, size=None)

        out = self.mlp((1 + self.eps) * x_original + aggregated_neighbors)

        out = self.norm(out)

        return out


class EdgeGNN(nn.Module):
    """
    Graph Neural Network model for edge-level prediction tasks,
    using stacked CustomGINConv layers.

    Args:
        in_channels (int): Dimensionality of input node features.
        hidden_channels (int): Dimensionality of hidden layers and final node embeddings.
        num_layers (int): Number of GIN layers.
        dropout_rate (float): Dropout probability applied between GNN layers.
        use_skip_connection (bool): If True, adds residual connections around GIN layers.
        use_layer_norm (bool): If True, applies Layer Normalization within GIN layers.
        use_bias (bool): If True, enables bias terms in GIN MLPs and predictor MLP.
        eps_trainable (bool): If True, the epsilon in GIN layers is learnable.
        task_type (str): Type of edge task ('link_prediction', 'edge_classification',
                         'edge_regression'). Determines the output head.
        num_edge_classes (int, optional): Number of classes for edge classification.
                                         Required if task_type='edge_classification'.
        use_input_norm (bool): If True, applies Layer Normalization to input features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout_rate: float = 0.5,
        use_skip_connection: bool = True,
        use_layer_norm: bool = True,
        use_bias: bool = True,
        eps_trainable: bool = True,
        task_type: str = "link_prediction",
        num_edge_classes: int | None = None,
        use_input_norm: bool = False,
    ):
        super().__init__()

        if task_type == "edge_classification" and num_edge_classes is None:
            raise ValueError(
                "num_edge_classes must be provided for task_type='edge_classification'"
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_skip_connection = use_skip_connection
        self.task_type = task_type
        self.num_edge_classes = num_edge_classes
        self.use_input_norm = use_input_norm

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.use_input_norm:
            self.input_norm = nn.LayerNorm(in_channels, elementwise_affine=True)
        else:
            self.input_norm = nn.Identity()

        current_dim = in_channels

        for i in range(num_layers):
            conv = CustomGINConv(
                in_channels=current_dim,
                out_channels=hidden_channels,
                mlp_hidden_channels=hidden_channels,
                train_eps=eps_trainable,
                use_layer_norm=use_layer_norm,
                use_bias=use_bias,
            )
            self.convs.append(conv)

            if use_skip_connection and i < num_layers - 1:
                self.norms.append(
                    nn.LayerNorm(hidden_channels, elementwise_affine=True)
                )
            else:
                self.norms.append(nn.Identity())

            current_dim = hidden_channels

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        predictor_input_dim = 2 * hidden_channels

        if task_type == "link_prediction":
            predictor_output_dim = 1
        elif task_type == "edge_classification":
            predictor_output_dim = num_edge_classes
        elif task_type == "edge_regression":
            predictor_output_dim = 1
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        self.edge_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_channels, bias=use_bias),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, predictor_output_dim, bias=use_bias),
        )

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """
        Computes node embeddings.

        Args:
            x (Tensor): Input node features [num_nodes, in_channels].
            edge_index (Adj): Graph connectivity [2, num_edges].

        Returns:
            Tensor: Final node embeddings [num_nodes, hidden_channels].
        """
        h = self.input_norm(x)

        for i in range(self.num_layers):
            h_in = h

            h = self.convs[i](h, edge_index)

            if i < self.num_layers - 1:
                if self.use_skip_connection:
                    h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

            if self.use_skip_connection and h_in.shape == h.shape:
                h = h + h_in

        return h

    def predict_edges(self, h: Tensor, edge_label_index: Adj) -> Tensor:
        """
        Predicts edge properties using final node embeddings.

        Args:
            h (Tensor): Final node embeddings [num_nodes, hidden_channels].
            edge_label_index (Adj): Edge indices for which to make predictions
                                    [2, num_prediction_edges].

        Returns:
            Tensor: Edge predictions (logits or values)
                    [num_prediction_edges, predictor_output_dim].
        """
        node_i_emb = h[edge_label_index[0]]
        node_j_emb = h[edge_label_index[1]]

        edge_feat = torch.cat([node_i_emb, node_j_emb], dim=-1)

        edge_predictions = self.edge_predictor(edge_feat)

        if self.task_type == "link_prediction" and edge_predictions.shape[-1] == 1:
            return edge_predictions.squeeze(-1)
        else:
            return edge_predictions
