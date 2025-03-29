import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM7b
from torch_geometric.data import DataLoader
from torch_geometric.nn import JumpingKnowledge, global_mean_pool
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.utils import degree

class GATLayerEdgeList(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1, alpha=0.2, dropout=0.6, bias=True):
        super(GATLayerEdgeList, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.alpha = alpha
        self.dropout = dropout
        
        self.W = nn.Parameter(torch.Tensor(in_features, n_heads * out_features))
        nn.init.xavier_uniform_(self.W)
        self.a = nn.Parameter(torch.Tensor(n_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_heads * out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        N = x.size(0)
        Wh = torch.matmul(x, self.W)
        Wh = Wh.view(N, self.n_heads, self.out_features)
        
        src, dst = edge_index
        Wh_src = Wh[src]
        Wh_dst = Wh[dst]
        
        a_input = torch.cat([Wh_src, Wh_dst], dim=-1)
        e = (a_input * self.a.unsqueeze(0)).sum(dim=-1)
        e = self.leakyrelu(e)
        
        e = e.transpose(0, 1)
        alpha_list = []
        for h in range(self.n_heads):
            alpha_h = softmax(e[h], dst, num_nodes=N)
            alpha_list.append(alpha_h)
        alpha = torch.stack(alpha_list, dim=0)
        alpha = alpha.transpose(0, 1)
        alpha = self.dropout_layer(alpha)
        
        alpha = alpha.unsqueeze(-1)
        messages = alpha * Wh_src
        messages = messages.reshape(-1, self.n_heads * self.out_features)
        
        h_prime = scatter_add(messages, dst, dim=0, dim_size=N)
        
        if self.bias is not None:
            h_prime = h_prime + self.bias
        
        h_prime = h_prime.view(N, self.n_heads, self.out_features)
        h_prime = h_prime.mean(dim=1)
        return h_prime

class GATEdgeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, n_heads=4, dropout=0.6):
        super(GATEdgeModel, self).__init__()
        self.device = device
        self.n_heads = n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        self.node_emb = nn.Linear(1, input_dim)
        self.dropout = dropout
        self.norm_input = nn.LayerNorm(input_dim)
        
        self.gat1 = GATLayerEdgeList(
            in_features=input_dim,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            bias=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.gat2 = GATLayerEdgeList(
            in_features=hidden_dim,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            bias=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.jump = JumpingKnowledge(mode='cat', channels=hidden_dim, num_layers=2)
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        neighbor_features = get_neighborhood_features(data).to(self.device)

        x = self.node_emb(neighbor_features)
        x = self.norm_input(x)

        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
        h1 = self.gat1(x, edge_index)
        h1 = F.relu(self.norm1(h1))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.gat2(h1, edge_index)
        h2 = F.relu(self.norm2(h2))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h = self.jump([h1, h2])

        pooled = global_mean_pool(h, data.batch)
        return self.mlp(pooled)

def get_neighborhood_features(data):
    row, col = data.edge_index
    deg = degree(col, data.num_nodes)
    neighbor_degrees = deg[row]
    mean_neighbor_degrees = scatter_add(neighbor_degrees, row, dim=0) / deg.clamp(min=1)
    return mean_neighbor_degrees.unsqueeze(1)


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

def compute_r2(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds = model(data)
            all_preds.append(preds)
            all_targets.append(data.y)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    ss_tot = ((all_targets - all_targets.mean(dim=0)) ** 2).sum()
    ss_res = ((all_targets - all_preds) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return r2.item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.NormalizeScale()
dataset = QM7b(root='./data/QM7b', pre_transform=transform)
#dataset = dataset.shuffle()
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

train_dataset = dataset[:n_train]
val_dataset = dataset[n_train:n_train+n_val]
test_dataset = dataset[n_train+n_val:]

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = GATEdgeModel(256,1024, 14, device, n_heads=4, dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()
epochs = 50
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) 

for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_error = evaluate(model, test_loader, criterion, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_error:.4f}')
    scheduler.step()

test_loss = evaluate(model, test_loader, criterion, device)
accuracy = compute_r2(model, test_loader, device)
print(f'Final Test Loss: {test_loss:.4f}, Final R²: {accuracy:.4f}')
