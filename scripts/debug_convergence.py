"""
scripts/debug_convergence.py
"""
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.loss.nll_loss import NLL_Loss
from models.transformer.transformer import Transformer_NN
from utils.data.motion_dataset import MotionDataset
from utils.viz.visualize_scenario import visualize_model_inputs_and_output

# --- CONFIGURATION ---
BATCH_SIZE = 8
LR = 0.0003  # Lowered from 0.01 (Transformers prefer lower LR)
EPOCHS = 200
LOG_FREQ = 10
PLOT_FREQ = 50
# Path to your data (adjust as needed for local vs colab)
DATA_PATH = "./data/uncompressed/tf_example/training/" 
# ---------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load ONLY one batch
print("Loading single batch...")
full_dataset = MotionDataset(DATA_PATH)
# We wrap it in a list to effectively 'freeze' this batch
single_batch = [full_dataset[i] for i in range(BATCH_SIZE)]
# Collate function is handled implicitly by the list of dicts structure if we manually stack,
# but using a DataLoader with size=total is easier if the dataset yields tensors.
# Since MotionDataset yields dicts of tensors, we can just use the first batch from the loader.
loader = DataLoader(full_dataset, batch_size=BATCH_SIZE)
fixed_batch = next(iter(loader))

# Move batch to device
for k, v in fixed_batch.items():
    if torch.is_tensor(v):
        fixed_batch[k] = v.to(device)

# 2. Initialize Model
print("Initializing model...")
# (Using dimensions extracted from the batch)
dummy_agent = fixed_batch['agent_input_continuous']
dummy_target = fixed_batch['agent_target']

model = Transformer_NN(
    num_agent_features=dummy_agent.shape[-1],
    num_static_road_features=fixed_batch['static_roadgraph_polyline_input'].shape[-1],
    num_dynamic_road_features=fixed_batch['dynamic_roadgraph_continuous'].shape[-1],
    num_past_timesteps=dummy_agent.shape[-2],
    num_model_features=256,
    categorical_embedding_dim=16,
    num_future_trajectories=3,
    num_future_timesteps=dummy_target.shape[-2],
    num_future_features=dummy_target.shape[-1]
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = NLL_Loss()

# Create debug output dir
os.makedirs("./tmp/debug_plots", exist_ok=True)

# 3. Training Loop
print(f"Starting overfitting on {BATCH_SIZE} samples...")
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    # Create random future agents input
    future_agents = torch.randn(
        (BATCH_SIZE, dummy_agent.shape[1], 3, 256),
        dtype=torch.float32, device=device)
    future_agents_valid = torch.ones(
        (BATCH_SIZE, dummy_agent.shape[1], 3), 
        dtype=torch.float32, device=device)

    # Forward Pass
    trajectories, probs = model(
        fixed_batch['agent_input_continuous'], 
        fixed_batch['agent_input_categorical'], 
        fixed_batch['agent_input_valid'],
        fixed_batch['static_roadgraph_polyline_input'], 
        fixed_batch['static_roadgraph_polyline_valid'],
        fixed_batch['dynamic_roadgraph_continuous'], 
        fixed_batch['dynamic_roadgraph_categorical'], 
        fixed_batch['dynamic_roadgraph_valid'],
        future_agents, 
        future_agents_valid
    )

    # --- CRITICAL CHANGE: Component-wise Loss Analysis ---
    # We calculate the internal terms of NLL_Loss manually here to see them
    ade_per_mode = loss_fn._compute_ade_per_trajectory(
        trajectories, 
        fixed_batch['agent_target'], 
        fixed_batch['agent_target_valid']
    )
    
    # 1. Geometry Loss (MinADE) - The most important one to watch first
    min_ade_loss = loss_fn.min_ade_loss(
        ade_per_mode, 
        fixed_batch['agent_target_valid'], 
        fixed_batch['tracks_to_predict']
    )
    
    # 2. Probability Loss (Weighted NLL)
    nll_loss = loss_fn.weighted_nll_loss(
        ade_per_mode, probs, 
        fixed_batch['agent_target_valid'], 
        fixed_batch['tracks_to_predict']
    )
    
    # 3. Diversity Loss
    div_loss = loss_fn.diversity_Loss(
        trajectories, 
        fixed_batch['agent_target_valid'], 
        fixed_batch['tracks_to_predict']
    )

    # Total Loss
    total_loss = min_ade_loss + nll_loss + div_loss

    total_loss.backward()
    optimizer.step()
    
    loss_history.append(total_loss.item())

    if epoch % LOG_FREQ == 0:
        print(f"Epoch {epoch} | Total: {total_loss.item():.4f} | "
              f"MinADE: {min_ade_loss.item():.4f} | "
              f"NLL: {nll_loss.item():.4f} | "
              f"Div: {div_loss.item():.4f}")

    # 4. Visualization Snapshot
    if epoch % PLOT_FREQ == 0 or epoch == EPOCHS - 1:
        print(f"Saving debug plot for epoch {epoch}...")
        model_output = {'agent_trajs': trajectories, 'agent_probs': probs}
        
        # We need to construct the input dict exactly as visualize_scenario expects
        # We detach tensors to avoid graph retention
        viz_input = {k: v.detach().clone() for k, v in fixed_batch.items()}
        # Rename keys to match visualization expectations if necessary
        viz_input['agent_input'] = viz_input.pop('agent_input_continuous')
        viz_input['static_roadgraph_input'] = viz_input.pop('static_roadgraph_polyline_input')
        viz_input['static_roadgraph_valid'] = viz_input.pop('static_roadgraph_polyline_valid')

        save_path = f"./tmp/debug_plots/epoch_{epoch:03d}.png"
        visualize_model_inputs_and_output(
            viz_input, 
            model_output, 
            index_in_batch=0, 
            should_visualize_outputs=True, 
            save_path=save_path
        )

print("Done! Check ./tmp/debug_plots to see the animation of convergence.")