import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM7b
from torch_geometric.data import DataLoader
from torch_geometric.nn import JumpingKnowledge, global_mean_pool
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.utils import degree

class GATLayerAdjMatrix(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1, alpha=0.2, dropout=0.6):
        super(GATLayerAdjMatrix, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.Tensor(in_features, n_heads * out_features))
        nn.init.xavier_uniform_(self.W)
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, adj):
        # x: (N, in_features), adj: (N, N)
        num_nodes = x.size(0)
        Wh = torch.matmul(x, self.W)  # (N, n_heads*out_features)
        Wh = Wh.view(num_nodes, self.n_heads, self.out_features)  # (N, n_heads, out_features)
        
        # Wh_i: (N, N, n_heads, out_features)
        Wh_i = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        # Wh_j: (N, N, n_heads, out_features)
        Wh_j = Wh.unsqueeze(0).repeat(num_nodes, 1, 1, 1)
        att_input = torch.cat([Wh_i, Wh_j], dim=-1)  # (N, N, n_heads, 2*out_features)
        att_input = att_input.view(num_nodes * num_nodes * self.n_heads, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(att_input, self.a))  # (N*N*n_heads, 1)
        e = e.view(num_nodes, num_nodes, self.n_heads)  # (N, N, n_heads)
        
        zero_vec = -1e10 * torch.ones_like(e)
        # Expand adj to (N, N, n_heads)
        adj_expanded = adj.unsqueeze(-1).expand_as(e)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        
        attention = attention.permute(2, 0, 1)       # (n_heads, N, N)
        Wh = Wh.permute(1, 0, 2)                     # (n_heads, N, out_features)
        h_prime = torch.bmm(attention, Wh)           # (n_heads, N, out_features)
        h_prime = h_prime.mean(dim=0)                # (N, out_features): average over heads
        return h_prime

class GATAdjModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, n_heads=4, dropout=0.6):
        super(GATAdjModel, self).__init__()
        self.node_emb = nn.Linear(1, input_dim)
        self.device = device
        
        self.gat1 = GATLayerAdjMatrix(input_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.gat2 = GATLayerAdjMatrix(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.jump = JumpingKnowledge(mode='cat', channels=hidden_dim, num_layers=2)
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data):
        neighbor_features = get_neighborhood_features(data).to(self.device)
        x = self.node_emb(neighbor_features)  # (N, input_dim)
        
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
        adj = to_dense_adj(edge_index, batch=data.batch, max_num_nodes=data.num_nodes)
        adj = adj.squeeze(0)  # (N, N) because batch size = 1
        
        h1 = self.gat1(x, adj)          # (N, hidden_dim)
        h1 = F.relu(self.norm1(h1))
        h1 = F.dropout(h1, p=0.6, training=self.training)
        
        h2 = self.gat2(h1, adj)          # (N, hidden_dim)
        h2 = F.relu(self.norm2(h2))
        h2 = F.dropout(h2, p=0.6, training=self.training)
        
        h = self.jump([h1, h2])          # (N, 2 * hidden_dim)
        pooled = global_mean_pool(h, data.batch)  # (B, 2 * hidden_dim); B=1 here
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
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader.dataset)

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

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = GATAdjModel(256,1024, 14, device, n_heads=4, dropout=0.5).to(device)
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
