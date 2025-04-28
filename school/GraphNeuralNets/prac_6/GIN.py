import os
import time
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

DATASET_NAME = 'Cora'
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
DROPOUT = 0.5
LEARNING_RATE = 0.01
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


path = os.path.join(os.getcwd(), 'data', DATASET_NAME)
dataset = Planetoid(root=path, name=DATASET_NAME)
data = dataset[0].to(DEVICE) 

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True)) 
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1: 
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1) 


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads=8):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
             self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
             x = F.dropout(x, p=self.dropout, training=self.training) 
             x = conv(x, edge_index)
             if i < len(self.convs) - 1: 
                 x = F.elu(x)
        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, train_eps=False):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList() 

        
        mlp1 = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            BN(hidden_channels), 
        )
        self.convs.append(GINConv(mlp1, train_eps=train_eps))
        self.bns.append(BN(hidden_channels)) 

        
        for _ in range(num_layers - 2):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                BN(hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.bns.append(BN(hidden_channels))

        
        mlp_out = Sequential(
            Linear(hidden_channels, hidden_channels), 
            ReLU(),
            Linear(hidden_channels, hidden_channels), 
            BN(hidden_channels),
        )
        self.convs.append(GINConv(mlp_out, train_eps=train_eps))
        self.bns.append(BN(hidden_channels))

        
        self.lin = Linear(hidden_channels, out_channels)


    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x) 
            x = F.relu(x)     
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lin(x) 
        return F.log_softmax(x, dim=1)




def train(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = {}
    for mask_name, mask in [('Train', data.train_mask), ('Val', data.val_mask), ('Test', data.test_mask)]:
        correct = pred[mask] == data.y[mask]
        accs[mask_name] = int(correct.sum()) / int(mask.sum())
    return accs




results = {}
models_to_test = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
    "GIN": GIN
}

for model_name, ModelClass in models_to_test.items():
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()

    model = ModelClass(
        in_channels=dataset.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=dataset.num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss() 

    best_val_acc = 0
    best_test_acc = 0
    history = {'loss': [], 'val_acc': [], 'test_acc': []}

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, optimizer, criterion, data)
        accs = test(model, data)
        train_acc, val_acc, test_acc = accs['Train'], accs['Val'], accs['Test']

        history['loss'].append(loss)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, ',
                  f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Finished training {model_name}. Time: {training_time:.2f}s")
    print(f"Best Validation Accuracy for {model_name}: {best_val_acc:.4f}")
    print(f"Test Accuracy at Best Validation for {model_name}: {best_test_acc:.4f}")
    results[model_name] = {'best_val_acc': best_val_acc, 'best_test_acc': best_test_acc, 'history': history, 'time': training_time}



print("\n--- Final Results ---")
print(f"Dataset: {DATASET_NAME}")
print("---------------------")
for model_name, result in results.items():
    print(f"{model_name}:")
    print(f"  Test Accuracy (at best validation): {result['best_test_acc']:.4f}")
    print(f"  Training Time: {result['time']:.2f}s")
print("---------------------")


plt.figure(figsize=(12, 8))


plt.subplot(2, 1, 1)
for model_name, result in results.items():
    plt.plot(range(1, EPOCHS + 1), result['history']['val_acc'], label=f'{model_name} Val Acc')
plt.title(f'{DATASET_NAME} - Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)


plt.subplot(2, 1, 2)
for model_name, result in results.items():
    plt.plot(range(1, EPOCHS + 1), result['history']['test_acc'], label=f'{model_name} Test Acc')
plt.title(f'{DATASET_NAME} - Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'{DATASET_NAME}_node_classification_comparison.png')
print(f"\nPlots saved to {DATASET_NAME}_node_classification_comparison.png")
plt.show()

print("\n--- Analysis ---")
print(f"Comparison on {DATASET_NAME}:")
print(" * GCN: Generally a strong baseline, performs well on homophilous datasets like Cora/CiteSeer.")
print(" * GAT: Often achieves slightly better performance than GCN by using attention mechanisms to weigh neighbors differently. Can be more computationally expensive.")
print(" * GraphSAGE: Designed for inductive learning (can generalize to unseen nodes/graphs), performs aggregation (mean, max, etc.). Performance varies based on dataset characteristics and aggregator.")
print(" * GIN: Theoretically the most powerful GNN among these in terms of distinguishing non-isomorphic graphs (related to the Weisfeiler-Lehman test). Its performance depends heavily on the expressiveness of the internal MLPs. It often requires careful tuning and may benefit from more layers or complex MLPs, especially on datasets where graph structure is highly informative beyond immediate neighbors.")
print("\nObservations from this run:")
for model_name, result in results.items():
     print(f" - {model_name} achieved a test accuracy of {result['best_test_acc']:.4f}.")