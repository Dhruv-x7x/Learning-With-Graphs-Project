import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
try:
    import torch
except ImportError:
    raise ImportError("You do not have torch installed. Run pip install torch in your terminal.")
import torch.nn as nn
import torch.optim as optim
from src.operations import create_graph, precompute_heat_kernels, plot_diffusion, find_paths, ReverseDiffusionModel, prepare_batch, compute_loss, beam_search_path, plot_path, lcs, dtw, is_valid_path

# DEFINE YOUR ROAD NETWORK DIMENSIONS
NUM_ROWS = 10
NUM_COLS= 10
EXTRA_EDGES = 25

# DEFINE DIFFUSION PARAMETERS
TVALUES = [0.1, 0.5, 1, 2, 5]

# DEFINE PATHS PARAMETERS
NUM_PATHS = 1000
MAX_PATH_LEN = 30

# DEFINE BEAM SEARCH PARAMETERS
BEAM_WIDTH = 50
BEAM_MAX_LEN = 30

# SEED
SEED = 42

def main():
    print("Running GDP... (Please change parameters directly in code rather than in shell)\n\n")
    # GENERATE RANDOM SEED
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # GENERATE GRAPH
    print("Generating graph...\n")
    G, pos = create_graph(n_rows=NUM_ROWS, n_cols=NUM_COLS, extra_edges=EXTRA_EDGES)
    num_nodes = G.number_of_nodes()
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    plt.title("Synthetic Road Network")
    plt.show()

    # PRECOMPUTE HEAT KERNELS
    print("Precomputing heat kernels...\n\n")
    t_values = TVALUES
    heat_kernels = precompute_heat_kernels(G, t_values,num_nodes)

    # PLOTTING DIFFUSION
    print("Plotting diffusion...\n\n")
    plot_diffusion(G,pos,0,heat_kernels,num_nodes,t_values)

    # GENERATING PATHS
    print("Generating paths...\n\n")
    paths = find_paths(G, NUM_PATHS, MAX_PATH_LEN)
    print(f"Generated {len(paths)} OD-path samples.")
    for ori, dst, path in paths:
        assert all(G.has_edge(path[i], path[i+1]) for i in range(len(path)-1)), f"Invalid path: {path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 16  

    # INTIALIZE MODEL AND OPTIMIZER
    print("Initializing model...\n\n")
    model = ReverseDiffusionModel(num_nodes, embedding_dim, device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TRAINING
    print("Starting training...\n\n")
    n_epochs = 1000
    for epoch in range(n_epochs):
        model.train()
        noisy_paths, clean_paths, oris, dsts, ts, masks, dst_pos = prepare_batch(paths, pos, t_values, device, batch_size=32)
        logits = model(noisy_paths, ts, oris, dsts, masks, pos, dst_pos)        
        loss = compute_loss(logits, clean_paths, masks, pos, num_nodes)        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    print("Training Complete.\n\n")

    # EXAMPLE PREDICTION
    print("Drawing a sample using beam search...\n\n")
    ori, dst, _ = random.choice(paths)
    planned_path = beam_search_path(model, G, ori, dst, pos, beam_width=25, max_len=20)
    print("Planned path:", planned_path)
    plot_path(G, pos, planned_path, color='r', label='Planned Path')

    # COMPUTING METRICS
    hits = []
    lcs_scores = []
    dtw_scores = []
    for i in range(20):
        ori, dst, gt_path = random.choice(paths)
        pred_path = beam_search_path(model, G, ori, dst, pos, beam_width=BEAM_WIDTH, max_len=BEAM_MAX_LEN)
        if not is_valid_path(G, pred_path):
            hits.append(False)
            continue
        hit = (pred_path[-1] == dst)
        hits.append(hit)
        lcs_scores.append(lcs(gt_path, pred_path))
        dtw_scores.append(dtw(gt_path, pred_path))
    
    print(f"Hit ratio: {np.mean(hits):.2f}, Avg LCS: {np.mean(lcs_scores):.2f}, Avg DTW: {np.mean(dtw_scores):.2f}")
    print("END\n")

if __name__ == "__main__":
    main()