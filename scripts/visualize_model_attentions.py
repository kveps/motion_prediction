"""
visualize_model_attentions.py — Per-head attention visualization for the
transformer motion prediction model.

Renders a 4-panel PNG (one panel per attention head) showing one of seven
attention surfaces for a single agent in a single scene.

Usage:
    python scripts/visualize_model_attentions.py \\
        --model-path models/trained_weights/best_model_k_1_e_8.pt \\
        --surface enc-agent-road \\
        --agent-idx 7 \\
        [--layer-idx -1] \\
        [--scene-idx 0] \\
        [--timestep-idx 10] \\
        [--mode-idx 0] \\
        [--top-k-polylines 20] \\
        [--local-data]
"""
import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.transformer.transformer import Transformer_NN
from utils.data.motion_dataset import MotionDataset


SCENE_SURFACES = {'enc-agent-self', 'enc-agent-road', 'enc-agent-traffic', 'dec-agent-self'}
MATRIX_SURFACES = {'enc-time-self', 'dec-mode-self', 'dec-mode-past'}
ALL_SURFACES = sorted(SCENE_SURFACES | MATRIX_SURFACES)

# Which optional args each surface actually consumes. Title rendering uses
# this so we don't print stale defaults (e.g. timestep=10 on enc-time-self).
SURFACE_USES_TIMESTEP = {'enc-agent-self', 'enc-agent-road', 'enc-agent-traffic'}
SURFACE_USES_MODE = {'dec-agent-self'}


def parse_args():
    p = argparse.ArgumentParser(
        description='Visualize attention weights of the transformer motion prediction model.')
    p.add_argument('--model-path', type=str, required=True,
                   help='Path to a checkpoint (.pt) — bundle or legacy weights-only.')
    p.add_argument('--surface', type=str, required=True, choices=ALL_SURFACES,
                   help='Which attention surface to visualize.')
    p.add_argument('--agent-idx', type=int, required=True,
                   help='Index of the agent whose attention to visualize (0..num_agents-1).')
    p.add_argument('--layer-idx', type=int, default=-1,
                   help='Layer index (default: -1 = last). Encoder and decoder each have 2 layers.')
    p.add_argument('--scene-idx', type=int, default=0,
                   help='Skip N scenes from the loader before sampling (default: 0).')
    p.add_argument('--timestep-idx', type=int, default=10,
                   help='Past timestep for encoder surfaces (default: 10 = current).')
    p.add_argument('--mode-idx', type=int, default=None,
                   help='Predicted mode for dec-agent-self. Default: argmax(prob_head).')
    p.add_argument('--top-k-polylines', type=int, default=20,
                   help='Only render top-K polylines by attention weight for enc-agent-road (default: 20).')
    p.add_argument('--local-data', action='store_true',
                   help='Read TFRecords from ./local_data/.')
    p.add_argument('--colab', action='store_true',
                   help='Read TFRecords from GCS bucket.')
    p.add_argument('--data-split', type=str, default='validation',
                   choices=['training', 'validation', 'testing'],
                   help='Which split to sample the scene from.')
    return p.parse_args()


def resolve_data_paths(args):
    if args.colab:
        return ('gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/training/',
                'gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/validation/',
                'gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/testing/')
    if args.local_data:
        return ('./local_data/training/', './local_data/validation/', './local_data/testing/')
    return ('./data/uncompressed/tf_example/training/',
            './data/uncompressed/tf_example/validation/',
            './data/uncompressed/tf_example/testing/')


def load_checkpoint_state(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt


def fetch_scene(loader, scene_idx):
    it = iter(loader)
    for _ in range(scene_idx):
        next(it)
    return next(it)


def extract_attention(model, surface, agent_idx, layer_idx, timestep_idx, mode_idx):
    """Returns the attention slice to render. Shape varies by surface."""
    enc_layers = model.transformer_encoder.layers
    dec_layers = model.transformer_decoder.layers

    if surface.startswith('enc-'):
        idx = layer_idx if layer_idx >= 0 else len(enc_layers) + layer_idx
        captured = enc_layers[idx].captured
        if surface == 'enc-time-self':
            return captured['time_self'][0, agent_idx]                              # [h, T, T]
        if surface == 'enc-agent-self':
            return captured['agent_self'][0, timestep_idx, :, agent_idx, :]         # [h, A]
        if surface == 'enc-agent-road':
            return captured['agent_static_road'][0, timestep_idx, :, agent_idx, :]  # [h, P]
        if surface == 'enc-agent-traffic':
            return captured['agent_dynamic_road'][0, timestep_idx, :, agent_idx, :] # [h, D]

    idx = layer_idx if layer_idx >= 0 else len(dec_layers) + layer_idx
    captured = dec_layers[idx].captured
    if surface == 'dec-agent-self':
        return captured['agent_self'][0, mode_idx, :, agent_idx, :]                 # [h, A]
    if surface == 'dec-mode-self':
        return captured['mode_self'][0, agent_idx]                                  # [h, K, K]
    if surface == 'dec-mode-past':
        return captured['mode_past'][0, agent_idx]                                  # [h, K, T]

    raise ValueError(f'Unknown surface: {surface}')


def build_title(surface, args):
    """Title that only mentions the parameters this surface actually consumes."""
    parts = [f'{surface}',
             f'layer={args.layer_idx}',
             f'agent={args.agent_idx}',
             f'scene={args.scene_idx}']
    if surface in SURFACE_USES_TIMESTEP:
        parts.append(f'timestep={args.timestep_idx}')
    if surface in SURFACE_USES_MODE:
        parts.append(f'mode={args.mode_idx}')
    return ' | '.join(parts)


def sqrt_normalize_per_plot(arr):
    """Sqrt scale, then divide by per-plot max so each panel uses its own range."""
    arr = np.sqrt(np.maximum(arr, 0.0))
    m = arr.max()
    if m > 0:
        arr = arr / m
    return arr


def render_matrix_panels(attention, surface, args, save_path, scene_xy_labels=None):
    """2x2 heatmap panels for matrix-style surfaces."""
    attn_np = attention.cpu().numpy()
    num_heads = attn_np.shape[0]
    # Axis labels per surface for clarity
    axis_labels = {
        'enc-time-self': ('key timestep', 'query timestep'),
        'dec-mode-self': ('key mode', 'query mode'),
        'dec-mode-past': ('key past timestep', 'query mode'),
    }
    xlabel, ylabel = axis_labels.get(surface, ('key index', 'query index'))

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()
    for h in range(4):
        ax = axes[h]
        if h < num_heads:
            mat = attn_np[h]
            n_rows, n_cols = mat.shape
            normed = sqrt_normalize_per_plot(mat)
            im = ax.imshow(normed, cmap='viridis', aspect='auto', origin='lower')
            ax.set_title(f'Head {h}')
            plt.colorbar(im, ax=ax)
            # Force integer ticks at every cell index — prevents matplotlib's
            # default tick locator from emitting decimal y-values for K=1 axes.
            ax.set_xticks(np.arange(n_cols))
            ax.set_yticks(np.arange(n_rows))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.axis('off')
    fig.suptitle(build_title(surface, args), fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# Styling constants — match utils/viz/visualize_scenario.py for visual consistency.
INPUT_COLOR = 'blue'
TARGET_COLOR = 'green'
OUTPUT_COLOR = 'red'
TRAJ_LW = 1.2
TRAJ_ALPHA = 0.8


def compute_viewport(agent_xy, agent_valid, target_xy, target_valid, agents_to_show):
    """Compute (x0, x1, y0, y1) viewport. Matches visualize_scenario.py:
    union of past + future positions of all salient agents, padded by 30m,
    minimum 40m extent."""
    pts = []
    for a in agents_to_show:
        v = agent_valid[a]
        if v.any():
            pts.append(np.stack([agent_xy[a, v, 0], agent_xy[a, v, 1]], axis=-1))
        tv = target_valid[a]
        if tv.any():
            pts.append(np.stack([target_xy[a, tv, 0], target_xy[a, tv, 1]], axis=-1))
    if not pts:
        return -80.0, 80.0, -80.0, 80.0
    pts = np.concatenate(pts, axis=0)
    cx = (pts[:, 0].max() + pts[:, 0].min()) / 2
    cy = (pts[:, 1].max() + pts[:, 1].min()) / 2
    half = max(pts[:, 0].max() - pts[:, 0].min(),
               pts[:, 1].max() - pts[:, 1].min(), 40.0) / 2 + 30.0
    return cx - half, cx + half, cy - half, cy + half


def _draw_road_base(ax, static_rg, static_rg_valid):
    """Match the alternating-shade road styling from visualize_scenario.py."""
    num_pl = static_rg.shape[0]
    for pl in range(num_pl):
        pv = static_rg_valid[pl]
        if pv.any():
            xs = static_rg[pl, pv, 0]
            ys = static_rg[pl, pv, 1]
            gray = 0.3 if pl % 2 == 0 else 0.5
            ax.plot(xs, ys, color=str(gray), linestyle='-', linewidth=0.8, alpha=0.6)


def _draw_agent_pasts(ax, agent_xy, agent_valid, agents_to_show):
    """Past trajectories (blue) + current-state square marker, per
    visualize_scenario.py conventions."""
    for a in agents_to_show:
        valid = agent_valid[a]
        if not valid.any():
            continue
        xs = agent_xy[a, valid, 0]
        ys = agent_xy[a, valid, 1]
        ax.plot(xs, ys, color=INPUT_COLOR, linewidth=TRAJ_LW, alpha=TRAJ_ALPHA)
        last = np.where(valid)[0][-1]
        ax.plot(agent_xy[a, last, 0], agent_xy[a, last, 1],
                marker='s', color='black', markersize=4, alpha=0.9, zorder=6)


def _draw_agent_targets(ax, target_xy, target_valid, agents_to_show):
    """Ground-truth future trajectories (green)."""
    for a in agents_to_show:
        valid = target_valid[a]
        if not valid.any():
            continue
        xs = target_xy[a, valid, 0]
        ys = target_xy[a, valid, 1]
        ax.plot(xs, ys, color=TARGET_COLOR, linewidth=TRAJ_LW, alpha=TRAJ_ALPHA)


def _draw_agent_predictions(ax, model_trajs, model_probs, tracks_to_predict, num_modes):
    """Model's most-probable predicted trajectory per tracks_to_predict agent (red)."""
    for a in np.where(tracks_to_predict)[0]:
        probs = model_probs[a]
        best_mode = int(np.argmax(probs))
        traj = model_trajs[a, best_mode, :, :2]
        ax.plot(traj[:, 0], traj[:, 1],
                color=OUTPUT_COLOR, linewidth=TRAJ_LW, alpha=TRAJ_ALPHA)


def _focus_agent_current_position(agent_xy, agent_valid, agent_idx):
    valid = agent_valid[agent_idx]
    if not valid.any():
        return None
    last = np.where(valid)[0][-1]
    return float(agent_xy[agent_idx, last, 0]), float(agent_xy[agent_idx, last, 1])


def _mark_focus_agent(ax, focus_xy):
    """Big red star over the focus agent's current position — attention-viz specific."""
    if focus_xy is None:
        return
    ax.plot(focus_xy[0], focus_xy[1], marker='*', color='red',
            markersize=22, markeredgecolor='black', markeredgewidth=1.0, zorder=20)


def render_scene_panels(attention, surface, args, scene, model_trajs, model_probs, save_path):
    """2x2 scene-overlay panels. One panel per head.

    Each panel draws (in z-order): road polylines → past trajectories (blue)
    → ground-truth target (green) → model prediction (red) → current-state
    markers → focus-agent star → attention overlay specific to this surface.
    """
    attn_np = attention.cpu().numpy()
    num_heads = attn_np.shape[0]

    # Scene tensors
    agent_xy = scene['agent_input_continuous'][0, :, :, :2].detach().cpu().numpy()
    agent_valid = scene['agent_input_valid'][0].detach().cpu().numpy().astype(bool)
    target_xy = scene['agent_target'][0, :, :, :2].detach().cpu().numpy()
    target_valid = scene['agent_target_valid'][0].detach().cpu().numpy().astype(bool)
    static_rg = scene['static_roadgraph_polyline_input'][0].detach().cpu().numpy()
    static_rg_valid = scene['static_roadgraph_polyline_valid'][0].detach().cpu().numpy().astype(bool)
    tracks_to_predict = scene['tracks_to_predict'][0].detach().cpu().numpy().astype(bool)
    is_sdc = (scene['is_sdc'][0].detach().cpu().numpy() > 0).astype(bool)
    dyn_xy = scene['dynamic_roadgraph_continuous'][0].detach().cpu().numpy()
    dyn_valid = scene['dynamic_roadgraph_valid'][0].detach().cpu().numpy().astype(bool)

    av_candidates = np.where(is_sdc)[0]
    av_idx = int(av_candidates[0]) if len(av_candidates) > 0 else None
    agents_to_show = set(np.where(tracks_to_predict)[0].tolist())
    if av_idx is not None:
        agents_to_show.add(av_idx)
    agents_to_show.add(args.agent_idx)

    focus_xy = _focus_agent_current_position(agent_xy, agent_valid, args.agent_idx)
    x0, x1, y0, y1 = compute_viewport(agent_xy, agent_valid, target_xy, target_valid, agents_to_show)

    # Convert model output to numpy for prediction overlay
    trajs_np = model_trajs[0].detach().cpu().numpy()
    probs_np = model_probs[0].detach().cpu().numpy()
    num_modes = trajs_np.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    cmap = plt.get_cmap('viridis')

    for h in range(4):
        ax = axes[h]
        if h >= num_heads:
            ax.axis('off')
            continue
        attn_h = attn_np[h]
        normed = sqrt_normalize_per_plot(attn_h)

        # 1) Road polylines (matching alternating-shade styling)
        _draw_road_base(ax, static_rg, static_rg_valid)
        # 1b) For enc-agent-road, overlay top-K polylines with colormap
        if surface == 'enc-agent-road':
            order = np.argsort(attn_h)
            top_idx = order[-args.top_k_polylines:][::-1]
            for pl in top_idx:
                pv = static_rg_valid[pl]
                if not pv.any() or attn_h[pl] <= 0:
                    continue
                xs = static_rg[pl, pv, 0]
                ys = static_rg[pl, pv, 1]
                w = normed[pl]
                ax.plot(xs, ys, color=cmap(w),
                        linewidth=1.5 + 2.5 * w, alpha=0.95, zorder=2)

        # 2a) Small marker for every valid agent so attention circles land on
        # visible entities, not apparent empty space.
        for a in range(agent_xy.shape[0]):
            if a in agents_to_show:
                continue  # will get the full blue-trajectory treatment below
            v = agent_valid[a]
            if not v.any():
                continue
            last = np.where(v)[0][-1]
            ax.plot(agent_xy[a, last, 0], agent_xy[a, last, 1],
                    marker='o', color='dimgrey', markersize=2.5, alpha=0.55, zorder=3)

        # 2b) Past trajectories (blue) + current-state markers for salient agents
        _draw_agent_pasts(ax, agent_xy, agent_valid, agents_to_show)

        # 3) Target trajectories (green)
        _draw_agent_targets(ax, target_xy, target_valid, agents_to_show)

        # 4) Predicted trajectories (red) for tracks_to_predict agents
        _draw_agent_predictions(ax, trajs_np, probs_np, tracks_to_predict, num_modes)

        # 5) Agent attention overlay (only for agent-self surfaces).
        # Size encodes attention weight; circle is uniformly black + semi-transparent
        # so it overlays cleanly on the scene without competing with trajectory colors.
        if surface in ('enc-agent-self', 'dec-agent-self'):
            for a in range(agent_xy.shape[0]):
                if a == args.agent_idx:
                    continue
                v = agent_valid[a]
                if not v.any() or attn_h[a] <= 0:
                    continue
                last = np.where(v)[0][-1]
                cx_a, cy_a = float(agent_xy[a, last, 0]), float(agent_xy[a, last, 1])
                w = normed[a]
                ax.plot(cx_a, cy_a, marker='o', color='black',
                        markersize=4 + 22 * w,
                        markeredgecolor='black', markeredgewidth=0.0,
                        alpha=0.35, zorder=7)

        # 6) Traffic lights — colored by attention for enc-agent-traffic, else faint context markers
        for d in range(dyn_xy.shape[0]):
            tv = dyn_valid[d]
            if not tv.any():
                continue
            last_t = np.where(tv)[0][-1]
            tx, ty = float(dyn_xy[d, last_t, 0]), float(dyn_xy[d, last_t, 1])
            if surface == 'enc-agent-traffic':
                w = normed[d]
                if attn_h[d] <= 0:
                    continue
                ax.plot(tx, ty, marker='^', color=cmap(w),
                        markersize=8 + 10 * w, markeredgecolor='black',
                        markeredgewidth=0.6, alpha=0.95, zorder=8)
            else:
                ax.plot(tx, ty, marker='^', color='goldenrod', markersize=5, alpha=0.4)

        # 7) Focus agent star on top
        _mark_focus_agent(ax, focus_xy)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Head {h}')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'{surface} | layer={args.layer_idx} agent={args.agent_idx} '
        f'scene={args.scene_idx} timestep={args.timestep_idx} mode={args.mode_idx}',
        fontsize=12, fontweight='bold',
    )

    # Shared scene-element legend, plus a surface-specific attention entry.
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=INPUT_COLOR, linewidth=TRAJ_LW, label='Past (input)'),
        Line2D([0], [0], color=TARGET_COLOR, linewidth=TRAJ_LW, label='Ground truth future'),
        Line2D([0], [0], color=OUTPUT_COLOR, linewidth=TRAJ_LW, label='Model prediction (best mode)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
               markersize=4, linestyle='none', label='Current state'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markeredgecolor='black', markersize=14, linestyle='none', label='Focus agent'),
        Line2D([0], [0], color='0.4', linewidth=0.8, label='Road polylines', alpha=0.6),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='dimgrey',
               markersize=4, linestyle='none', alpha=0.55,
               label='Other agent (any valid past frame)'),
    ]
    if surface in ('enc-agent-self', 'dec-agent-self'):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=14, alpha=0.35, linestyle='none',
                   label='Attention to other agent (size ∝ weight)')
        )
    elif surface == 'enc-agent-road':
        legend_elements.append(
            Line2D([0], [0], color=plt.get_cmap('viridis')(0.9), linewidth=2.6,
                   label=f'Top-{args.top_k_polylines} attended polyline (color/thickness ∝ weight)')
        )
    elif surface == 'enc-agent-traffic':
        legend_elements.append(
            Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=plt.get_cmap('viridis')(0.9),
                   markeredgecolor='black', markersize=14, linestyle='none',
                   label='Attended traffic light (color/size ∝ weight)')
        )
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9, framealpha=0.95)

    fig.tight_layout(rect=[0, 0.03, 1, 1.0])
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading checkpoint: {args.model_path}')
    state_dict = load_checkpoint_state(args.model_path, device)
    num_modes = state_dict['mode_queries.weight'].shape[0]
    print(f'Detected K (num_future_trajectories) = {num_modes}')

    training_path, validation_path, testing_path = resolve_data_paths(args)
    split_paths = {'training': training_path, 'validation': validation_path, 'testing': testing_path}
    data_path = split_paths[args.data_split]
    print(f'Loading scene from: {data_path}')
    dataset = MotionDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    scene = fetch_scene(loader, args.scene_idx)
    print(f'Got scene index {args.scene_idx}')

    # Infer model input/output shapes from this scene
    agent_in = scene['agent_input_continuous']
    static_in = scene['static_roadgraph_polyline_input']
    dyn_in = scene['dynamic_roadgraph_continuous']
    target = scene['agent_target']

    model = Transformer_NN(
        num_agent_features=agent_in.size(-1),
        num_static_road_features=static_in.size(-1),
        num_dynamic_road_features=dyn_in.size(-1),
        num_past_timesteps=agent_in.size(-2),
        num_model_features=256,
        categorical_embedding_dim=16,
        num_future_trajectories=num_modes,
        num_future_timesteps=target.size(-2),
        num_future_features=target.size(-1),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Forward pass with capture
    agents_cont = scene['agent_input_continuous'].to(device)
    agents_cat = scene['agent_input_categorical'].to(device)
    agents_valid = scene['agent_input_valid'].to(device)
    static_road = scene['static_roadgraph_polyline_input'].to(device)
    static_road_valid = scene['static_roadgraph_polyline_valid'].to(device)
    dyn_cont = scene['dynamic_roadgraph_continuous'].to(device)
    dyn_cat = scene['dynamic_roadgraph_categorical'].to(device)
    dyn_valid = scene['dynamic_roadgraph_valid'].to(device)
    with torch.no_grad():
        trajectories, probs = model(
            agents_cont, agents_cat, agents_valid,
            static_road, static_road_valid,
            dyn_cont, dyn_cat, dyn_valid,
            capture=True,
        )

    # Resolve mode_idx for dec-agent-self if not given
    if args.surface == 'dec-agent-self' and args.mode_idx is None:
        args.mode_idx = int(probs[0, args.agent_idx].argmax().item())
        print(f'Resolved --mode-idx = {args.mode_idx} (argmax of prob_head)')

    attn = extract_attention(
        model, args.surface, args.agent_idx, args.layer_idx,
        args.timestep_idx, args.mode_idx,
    )
    print(f'Extracted attention tensor shape: {tuple(attn.shape)}')

    os.makedirs('./tmp', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = (f'./tmp/attention_{args.surface}_layer{args.layer_idx}_'
             f'agent{args.agent_idx}_scene{args.scene_idx}_{ts}.png')

    if args.surface in MATRIX_SURFACES:
        render_matrix_panels(attn, args.surface, args, fname)
    else:
        render_scene_panels(attn, args.surface, args, scene, trajectories, probs, fname)

    print(f'Saved: {fname}')


if __name__ == '__main__':
    main()
