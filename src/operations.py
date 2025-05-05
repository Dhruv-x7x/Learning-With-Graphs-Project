import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
try:
    import torch
except ImportError:
    raise ImportError("You do not have torch installed. Run pip install torch in your terminal.")
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import expm
from collections import deque
import random
from difflib import SequenceMatcher
from torch.utils.data import DataLoader
import torch.nn.functional as F
try:
    import einops
except ImportError:
    raise ImportError("You do not have einops installed. Run pip install einops in your terminal.")
from einops.layers.torch import Rearrange
from einops import rearrange

def create_graph(n_rows=4, n_cols=5, extra_edges=3):
    G = nx.grid_2d_graph(n_rows, n_cols)
    
    # Relabel to integer nodes
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    pos = {i: node for node, i in mapping.items()}
    
    nodes = list(G.nodes())
    added = 0
    attempts = 0
    max_attempts = extra_edges * 10  # prevent infinite loop

    while added < extra_edges and attempts < max_attempts:
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v) and abs(u - v) != 1 and abs(u - v) != n_cols:
            G.add_edge(u, v)
            added += 1
        attempts += 1
    return G, pos

class SinusoidalPosEmb(nn.Module):
    # This class was taken directly from the author's own implementation
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device
        inv_freq = torch.exp(-torch.arange(0, dim, 2) * np.log(10000) / dim).to(device)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        """
        t: Tensor of shape (batch,) - timestep values
        returns: Tensor of shape (batch, dim)
        """
        sinusoid_inp = t[:, None] * self.inv_freq[None, :]  # (B, D//2)
        emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return emb  # (B, D)

class GDPEmbedding(nn.Module):
    def __init__(self, num_nodes, embed_dim, device):
        super().__init__()
        self.phi = nn.Embedding(num_nodes, embed_dim)  # Learnable node embeddings 
        self.time_embedding = SinusoidalPosEmb(embed_dim, device)  # Sinusoidal timestep embedding so that the diffusion model has a sense of time

    def forward(self, node_ids, timesteps):
        """
        node_ids: Tensor of shape (batch_size, path_len), integers for each node in the path
        timesteps: Tensor of shape (batch_size,) with diffusion step index
        """
        x_embed = self.phi(node_ids)            # (B, L, D)
        t_embed = self.time_embedding(timesteps)  # (B, D)

        return x_embed, t_embed
    
def precompute_heat_kernels(G, t_values, num_nodes):
    A = nx.to_numpy_array(G)
    D = np.diag([G.degree(i) for i in range(num_nodes)])
    L = A - D
    kernels = {}
    for t in t_values:
        C = expm(t * L)
        C = C / C.sum(axis=1, keepdims=True)
        kernels[t] = C
    return kernels

def plot_diffusion(G, pos, node, heat_kernels, num_nodes, steps=[0.1, 1, 2, 5]):
    fig, axs = plt.subplots(1, len(steps), figsize=(15, 3))
    for i, t in enumerate(steps):
        Ct = heat_kernels[t]
        p = np.zeros(num_nodes)
        p[node] = 10
        p = p @ Ct
        nx.draw(G, pos, node_color=p, cmap='hot', vmin=0, vmax=1, ax=axs[i], with_labels=True)
        axs[i].set_title(f"t={t}")
    plt.suptitle(f"Diffusion from node {node}")
    plt.show()

def find_paths(G, num_paths=100, max_length=10):
    nodes = list(G.nodes())
    paths = []
    for _ in range(num_paths):
        while True:
            ori, dst = np.random.choice(nodes, 2, replace=False) # select any 2 nodes to be our origin and destination
            try:
                # Always start with a valid shortest path
                path = nx.shortest_path(G, ori, dst)
                # Randomly add a detour at a random position
                if len(path) > 3 and np.random.rand() < 0.3:
                    idx = np.random.randint(1, len(path)-1)
                    prev = path[idx-1]
                    curr = path[idx]
                    # Only add a neighbor of curr that is not already in the path
                    neighbors = [n for n in G.neighbors(curr) if n not in path]
                    if neighbors:
                        detour = np.random.choice(neighbors)
                        # Insert detour between curr and next, ensuring connectivity
                        path = path[:idx+1] + [detour] + path[idx+1:]
                        # Ensure the detour is connected to both curr and next
                        if not (G.has_edge(curr, detour) and G.has_edge(detour, path[idx+1])):
                            continue  # Skip if not actually connected
                # Validate path connectivity
                if 3 < len(path) <= max_length and all(G.has_edge(path[i], path[i+1]) for i in range(len(path)-1)):
                    paths.append((ori, dst, path))
                    break
            except nx.NetworkXNoPath:
                continue
    return paths

def diffuse_path(path, t, heat_kernels, num_nodes):
    C = heat_kernels[t]
    noisy_path = []
    for node in path:
        p = np.zeros(num_nodes)
        p[node] = 1
        p = p @ C
        noisy_node = np.random.choice(num_nodes, p=p)
        noisy_path.append(noisy_node)
    return noisy_path

class LinearAttention(nn.Module):
    def __init__(self, dim, device, heads=4, dim_head=32):
        super().__init__()
        self.device = device
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False, device=device)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1, device=device)

    def forward(self, x, masks):
        # x shape is expected to be [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.shape
        
        # Transpose x for 1D convolution operations
        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
        
        # Apply mask if provided
        if masks is not None:
            # Expand mask for broadcasting
            expanded_mask = masks.unsqueeze(1).expand(-1, embed_dim, -1)
            x = x.masked_fill(expanded_mask, 0)
        
        # QKV projections
        qkv = self.to_qkv(x).chunk(3, dim=1)
        
        # Reshape for multi-head attention
        # Original shape after conv: [batch, heads*dim_head, seq_len]
        # Target shape: [batch, heads, dim_head, seq_len]
        q, k, v = map(lambda t: t.reshape(batch_size, self.heads, -1, seq_len), qkv)
        
        # Apply scaling
        q = q * self.scale
        
        # Apply masks to q and k if provided
        if masks is not None:
            # Reshape mask for broadcasting with q and k
            attn_mask = masks.unsqueeze(1).unsqueeze(2).expand(-1, self.heads, 1, -1)
            q = q.masked_fill(attn_mask, 0)
            k = k.masked_fill(attn_mask, -1e15)
        
        # Compute attention weights and apply to values
        # k shape: [batch, heads, dim_head, seq_len]
        k = k.softmax(dim=-1)  # Softmax along sequence dimension
        
        # Matrix multiplication: k.transpose(-2, -1) @ v
        # [batch, heads, dim_head, seq_len] x [batch, heads, dim_head, seq_len] -> [batch, heads, dim_head, dim_head]
        context = torch.matmul(k, v.transpose(-2, -1))  # Replace einsum with matmul
        
        # Matrix multiplication: context @ q
        # [batch, heads, dim_head, dim_head] x [batch, heads, dim_head, seq_len] -> [batch, heads, dim_head, seq_len]
        out = torch.matmul(context, q)  # Replace einsum with matmul
        
        # Reshape back: [batch, heads, dim_head, seq_len] -> [batch, heads*dim_head, seq_len]
        out = out.reshape(batch_size, -1, seq_len)
        
        # Project back to original dimension
        out = self.to_out(out)
        
        # Transpose back to original shape: [batch, embed_dim, seq_len] -> [batch, seq_len, embed_dim]
        return out.permute(0, 2, 1)
    
class ReverseDiffusionModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim=64, n_heads=2, device='cpu'):
        super().__init__()
        self.device = device
        # Use the GDPEmbedding for node and temporal embeddings
        self.embedding = GDPEmbedding(num_nodes, embedding_dim, device)  # Using GDPEmbedding class
        self.t_embed = nn.Linear(1, hidden_dim)  # Linear transformation of time step
        self.OD_embed = nn.Linear(2 * embedding_dim, hidden_dim)  # Origin-Destination embedding
        self.spatial_fc = nn.Linear(2, hidden_dim)  # Spatial features: distance and direction
        
        # Replace MultiHeadAttention with LinearAttention for better scalability
        self.attn = LinearAttention(embedding_dim, device, heads=n_heads, dim_head=hidden_dim)
        
        # Convolutional layers for processing path representations
        self.conv1 = nn.Conv1d(embedding_dim + hidden_dim * 3, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Fully connected layer to predict logits for the nodes
        self.fc = nn.Linear(hidden_dim, num_nodes)

    def forward(self, noisy_path, t, ori, dst, masks, pos, dst_pos):
        batch_size, path_len = noisy_path.shape
        
        # Get node embeddings for the noisy path using GDPEmbedding
        node_embs, _ = self.embedding(noisy_path, t)  # Output shape: (batch_size, path_len, embedding_dim)
        
        # Apply linear attention to the node embeddings
        attn_out = self.attn(node_embs, masks)  # Output shape: (batch_size, path_len, embedding_dim)
        
        # Compute spatial features (distance and direction) for each node in the path
        spatial_feats = []
        for b in range(batch_size):
            feats = []
            for i in range(path_len):
                # Skip if mask is True (indicating padding)
                if masks[b, i]:
                    feats.append([0.0, 0.0])
                    continue
                    
                n = noisy_path[b, i].item()
                node_xy = np.array(pos[n])
                dst_xy = np.array(dst_pos[b])
                
                # Calculate distance and direction
                dist = np.linalg.norm(node_xy - dst_xy)
                direction = 0.0
                if dist > 1e-6:
                    direction = np.dot((dst_xy - node_xy), [1, 0]) / dist
                    
                feats.append([dist, direction])
            spatial_feats.append(feats)
        
        # Convert spatial features to a tensor
        spatial_feats = torch.tensor(spatial_feats, dtype=torch.float32, device=node_embs.device)
        
        # Pass spatial features through a fully connected layer
        spatial_embs = self.spatial_fc(spatial_feats)
        
        # Get the timestep embedding
        t_emb = self.t_embed(t.view(-1, 1)).unsqueeze(1).repeat(1, path_len, 1)
        
        # Get the Origin-Destination embedding
        OD_emb = self.OD_embed(torch.cat([self.embedding.phi(ori), self.embedding.phi(dst)], dim=-1)).unsqueeze(1).repeat(1, path_len, 1)
        
        # Concatenate all embeddings
        x = torch.cat([attn_out, t_emb, OD_emb, spatial_embs], dim=-1)
        
        # Permute for convolution
        x = x.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        
        # Apply convolution layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Permute back
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        
        # Final projection to get logits
        logits = self.fc(x)
        
        return logits
    

def prepare_batch(paths, pos, t_values, device, batch_size=16):
    batch = random.sample(paths, batch_size)
    max_len = max(len(p[2]) for p in batch)
    noisy_paths, clean_paths, oris, dsts, ts, masks, dst_pos = [], [], [], [], [], [], []
    
    for ori, dst, path in batch:
        t = np.random.choice(t_values)  # Random timestep
        noisy = diffuse_path(path, t)  # Diffuse the path
        
        path_len = len(path)
        pad_len = max_len - path_len
        
        # Pad paths with zeros
        noisy_padded = noisy + [0] * pad_len
        path_padded = path + [0] * pad_len
        
        # Create mask: False (0) for valid positions, True (1) for padding positions
        mask = [False] * path_len + [True] * pad_len
        
        # Enforce O/D at endpoints
        noisy_padded[0] = ori
        noisy_padded[path_len-1] = dst
        path_padded[0] = ori
        path_padded[path_len-1] = dst
        
        # Do not compute loss for origin and destination
        mask[0] = True  # Mask loss for origin
        mask[path_len-1] = True  # Mask loss for destination
        
        noisy_paths.append(noisy_padded)
        clean_paths.append(path_padded)
        oris.append(ori)
        dsts.append(dst)
        ts.append(t)
        masks.append(mask)
        dst_pos.append(pos[dst])
    
    return (torch.LongTensor(noisy_paths).to(device),
            torch.LongTensor(clean_paths).to(device),
            torch.LongTensor(oris).to(device),
            torch.LongTensor(dsts).to(device),
            torch.FloatTensor(ts).to(device),
            torch.BoolTensor(masks).to(device),  # Using BoolTensor for clarity
            dst_pos)

def compute_loss(logits, clean_paths, masks, adjacency_matrix, num_nodes):
    """
    Compute KL divergence loss, cross-entropy loss, and contribution loss.
    """
    batch_size, seq_len = logits.shape[0], logits.shape[1]
    
    # Cross-entropy loss (masked)
    clean_paths = clean_paths.view(-1)
    masks_flat = masks.view(-1)
    logits_flat = logits.view(-1, num_nodes)
    
    ce_loss = nn.CrossEntropyLoss()(logits_flat[masks_flat], clean_paths[masks_flat])
    
    # Contribution loss (penalize invalid transitions according to the adjacency matrix)
    eps = 1e-6  # Small epsilon to avoid numerical issues
    node_probs = torch.softmax(logits_flat, dim=-1)
    
    # Check if adjacency_matrix is a dict and convert it to a tensor if needed
    if isinstance(adjacency_matrix, dict):
        # Create a tensor from the adjacency matrix dictionary
        # This is an example approach - you might need to adjust based on your specific dict structure
        adj_tensor = torch.zeros_like(node_probs)
        
        # Populate the adjacency tensor based on your dictionary structure
        # Example (adjust according to your actual dictionary structure):
        for i in range(adj_tensor.shape[0]):
            for j in range(adj_tensor.shape[1]):
                node_id = i % num_nodes  # Get the node ID
                if node_id in adjacency_matrix:
                    # Assuming adjacency_matrix[node_id] contains valid transitions
                    if j in adjacency_matrix[node_id]:
                        adj_tensor[i, j] = 1.0
                        
        # Use the tensor for the calculation
        con_loss = -torch.sum(torch.log(node_probs + eps) * adj_tensor) / batch_size
    else:
        # Assuming adjacency_matrix is already a tensor with the right shape
        con_loss = -torch.sum(torch.log(node_probs + eps) * adjacency_matrix) / batch_size
    
    # Return total loss
    return ce_loss + con_loss

def beam_search_path(model, G, ori, dst, pos, beam_width=5, max_len=20, t=1.0):
    model.eval()
    beams = [([ori], 0.0)]  # Each beam is (path, cumulative_log_prob)
    dst_pos = [pos[dst]]
    
    for step in range(max_len - 1):
        new_beams = []
        for path, log_prob in beams:
            current = path[-1]
            
            # Only allow neighbors (and destination if it's a neighbor)
            neighbors = list(G.neighbors(current))
            if dst not in neighbors and current != dst:
                candidates = neighbors
            else:
                candidates = neighbors + ([dst] if dst in neighbors else [])
            
            if not candidates:
                continue  # Dead end
                
            # Diffuse the path for the noisy version with reduced noise
            noisy_path = diffuse_path(path, t=0.5)  # Smaller value for t reduces noise
            pad_len = max_len - len(noisy_path)
            noisy_path_tensor = torch.LongTensor([noisy_path + [0] * pad_len]).to(model.device)
            
            # Create the mask for valid positions (1 for valid node, 0 for padding)
            mask = torch.BoolTensor([[True] * len(noisy_path) + [False] * pad_len]).to(model.device)
            
            # Prepare the other necessary inputs
            ori_tensor = torch.LongTensor([ori]).to(model.device)
            dst_tensor = torch.LongTensor([dst]).to(model.device)
            t_tensor = torch.FloatTensor([t]).to(model.device)
            
            with torch.no_grad():
                # Perform the forward pass of the model
                logits = model(noisy_path_tensor, t_tensor, ori_tensor, dst_tensor, mask, pos, dst_pos)
                
                # Extract the logits for the current step (corresponding to the last node in the path)
                current_step_logit = logits[0, len(path) - 1]
                
                # Convert logits to probabilities using softmax
                probs = torch.softmax(current_step_logit, dim=-1).cpu().numpy()
            
            # Only consider valid next nodes (candidates)
            for n in candidates:
                prob = probs[n]
                if prob > 0:  # Consider only non-zero probability nodes
                    new_path = path + [n]
                    
                    # Ensure no excessive repetition of nodes in the path
                    if new_path.count(n) >= 3:  # Skip paths with too many repetitions
                        continue
                    
                    new_log_prob = log_prob + np.log(prob)
                    new_beams.append((new_path, new_log_prob))
        
        # Prune to top beam_width beams by cumulative log probability
        new_beams = sorted(new_beams, key=lambda x: -x[1])[:beam_width]
        beams = new_beams
        
        # If any beam ends at dst, return the first such path
        for path, _ in beams:
            if path[-1] == dst:
                return path
        
        if not beams:
            break  # No valid extensions
        
    # If no path reaches dst, return the best partial beam
    return beams[0][0] if beams else [ori]

def plot_path(G, pos, path, color='r', label='Planned'):
    # Draw the graph with node labels and positions
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12)
    
    # Create edges from the planned path
    edges = list(zip(path[:-1], path[1:]))  # Create a list of edges from the path
    
    # Draw the edges of the path in the specified color
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=2, label=label)
    
    # Display the plot with a title
    plt.title(label)
    plt.show()

def lcs(a, b):
    # Find the longest common subsequence between two paths
    matcher = SequenceMatcher(None, a, b)
    return matcher.find_longest_match(0, len(a), 0, len(b)).size

def dtw(a, b):
    # Calculate Dynamic Time Warping distance between two paths
    n, m = len(a), len(b)
    dp = np.full((n+1, m+1), float('inf'))
    dp[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m]

def is_valid_path(G, path):
    # Check if the path is valid in the graph (all consecutive edges must exist)
    return all(G.has_edge(path[i], path[i+1]) for i in range(len(path)-1))