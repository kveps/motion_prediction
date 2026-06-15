"""Transformer training / testing entrypoint.

Supports three execution environments:

  * Local with on-disk TFRecords  (default)
  * Local with preprocessed .pt   (--preprocessed)
  * Google Colab over gcsfuse     (--colab)

Plus a one-shot test/visualization mode (--test --model-path <ckpt>).

Multi-modal anchor configuration sits at the very top of this file (see the
"DECODER ANCHOR CONFIG" block) — change ANCHOR_TYPE / NUM_MODES there. If a
centroid-based anchor is selected and the centroids file is missing, it is
computed automatically from the training set on first run.
"""
from models.loss.nll_loss import NLL_Loss
from models.transformer.transformer import Transformer_NN
from utils.data.motion_dataset import MotionDataset, PreprocessedMotionDataset
from utils.model.endpoint_anchors import (
    ANCHOR_BALLISTIC,
    ANCHOR_CENTROID,
    ANCHOR_HYBRID,
    VALID_ANCHOR_TYPES,
    compute_and_save_centroids,
    load_centroids,
)
from utils.viz.visualize_scenario import visualize_model_inputs_and_output
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import datetime
import argparse
import os


# =========================================================================
# DECODER ANCHOR CONFIG
# =========================================================================
# The decoder needs an initial 2D endpoint per (agent, mode). These three
# options control where that anchor comes from. See
# utils/model/endpoint_anchors.py for full implementations.
#
#   ANCHOR_BALLISTIC : K different yaw rates applied to each agent's
#                      current state. Self-contained — no offline data.
#                      Diversity vanishes for stopped agents.
#
#   ANCHOR_CENTROID  : K-means cluster centers of training-set endpoints
#                      (agent-local frame), rotated/translated per agent at
#                      runtime. Same K maneuvers for every agent; provides
#                      stable semantic mode identity. Requires CENTROIDS_PATH.
#
#   ANCHOR_HYBRID    : Per-agent constant-velocity ballistic endpoint +
#                      per-mode centroid offset. Combines per-agent motion
#                      with per-mode maneuver intent. Requires CENTROIDS_PATH.
#
# For CENTROID and HYBRID: if CENTROIDS_PATH does not exist, the script
# computes the centroids from the training dataset on first run and saves
# them.
# =========================================================================
ANCHOR_TYPE     = ANCHOR_HYBRID
NUM_MODES       = 6                    # K — decoder mode count. MTR uses 6.
NUM_CENTROIDS   = 6                    # >= NUM_MODES. Only used by CENTROID/HYBRID.
CENTROIDS_PATH  = './models/trained_weights/intention_centroids.pt'

assert ANCHOR_TYPE in VALID_ANCHOR_TYPES
assert NUM_CENTROIDS >= NUM_MODES, \
    f"NUM_CENTROIDS ({NUM_CENTROIDS}) must be >= NUM_MODES ({NUM_MODES})."


if __name__ == '__main__':
    # =====================================================================
    # Argument parsing
    # =====================================================================
    parser = argparse.ArgumentParser(description='Train or test Transformer model')
    parser.add_argument('--colab', action='store_true',
                        help='Use Google Colab mode with GCS paths (requires authentication)')
    parser.add_argument('--local-data', action='store_true',
                        help='Read TFRecords from ./local_data/ instead of the gcsfuse mount at ./data/')
    parser.add_argument('--test', action='store_true',
                        help='Run in testing mode instead of training')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model weights for testing')
    parser.add_argument('--epochs', type=int, default=32,
                        help='Number of training epochs (default: 32)')
    parser.add_argument('--batch-size', type=int, default=48,
                        help='Batch size for training/validation (default: 48)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--num-train-files', type=int, default=600,
                        help='Number of training TFRecord files to use per epoch (default: 600). '
                             'Files are reshuffled each epoch so the full dataset is covered over time.')
    parser.add_argument('--preprocessed', action='store_true',
                        help='Use preprocessed .pt files instead of TFRecords (much faster)')
    parser.add_argument('--preprocessed-root', type=str, default='./data/preprocessed',
                        help='Root directory of preprocessed .pt files (default: ./data/preprocessed)')
    parser.add_argument('--data-split', type=str, default='validation',
                        choices=['training', 'validation', 'testing'],
                        help='Data split to use in test mode (default: validation). '
                             'Note: the testing split has no ground truth, so loss will be 0.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from a checkpoint. Accepts a full bundle '
                             '(model + optimizer + epoch + best_val_loss) or a legacy '
                             'weights-only .pt (warm-starts model, resets optimizer/epoch).')
    parser.add_argument('--warm-start', type=str, default=None,
                        help='Load shape-compatible weights from a checkpoint and start '
                             'training from epoch 0 with a fresh optimizer. Parameters '
                             'whose names/shapes differ (e.g. mode_queries when K changes) '
                             'keep their random init. Use when transferring between configs '
                             '(e.g. K=1 baseline -> K=6 multimodal).')
    parser.add_argument('--centroids-samples', type=int, default=100_000,
                        help='Cap on number of training endpoints used for k-means when '
                             'computing centroids (default: 100k). Only consulted if the '
                             'centroids file at CENTROIDS_PATH does not yet exist.')
    args = parser.parse_args()

    # =====================================================================
    # Device + path config
    # =====================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.colab:
        print("Running in Colab mode - using GCS paths")
        TRAINING_PATH   = "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/training/"
        VALIDATION_PATH = "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/validation/"
        TESTING_PATH    = "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/testing/"

        if os.path.exists("/content/drive/MyDrive"):
            SAVE_DIR = "/content/drive/MyDrive/av_prediction/models/trained_weights/"
            print("✓ Using Google Drive for model storage")
        else:
            SAVE_DIR = "/content/models/trained_weights/"
            print("⚠ Google Drive not mounted - saving to /content/ (temporary storage)")
    elif args.local_data:
        print("Running in local-data mode (./local_data/)")
        TRAINING_PATH   = "./local_data/training/"
        VALIDATION_PATH = "./local_data/validation/"
        TESTING_PATH    = "./local_data/testing/"
        SAVE_DIR        = "./models/trained_weights/"
    else:
        print("Running in local mode (gcsfuse mount)")
        TRAINING_PATH   = "./data/uncompressed/tf_example/training/"
        VALIDATION_PATH = "./data/uncompressed/tf_example/validation/"
        TESTING_PATH    = "./data/uncompressed/tf_example/testing/"
        SAVE_DIR        = "./models/trained_weights/"

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Models will be saved to: {SAVE_DIR}")

    # =====================================================================
    # Dataset + dataloader construction
    # =====================================================================
    print("Loading datasets...")
    if args.test:
        split_paths = {'training': TRAINING_PATH,
                       'validation': VALIDATION_PATH,
                       'testing': TESTING_PATH}
        eval_path = split_paths[args.data_split]
        print(f"Test mode using '{args.data_split}' split: {eval_path}")
        if args.data_split == 'testing':
            print("Warning: testing split has no ground truth — loss will be 0.")
        test_dataset    = MotionDataset(eval_path)
        test_dataloader = DataLoader(test_dataset, batch_size=1)
        dummy_element   = test_dataset[0]
        training_dataset = None  # not used in test mode
    elif args.preprocessed:
        print("Using preprocessed .pt datasets")
        training_dataset   = PreprocessedMotionDataset(
            os.path.join(args.preprocessed_root, 'training'))
        validation_dataset = PreprocessedMotionDataset(
            os.path.join(args.preprocessed_root, 'validation'))
        training_dataloader   = DataLoader(training_dataset, batch_size=args.batch_size,
                                           num_workers=4, pin_memory=True,
                                           persistent_workers=True,
                                           prefetch_factor=4)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size,
                                           num_workers=4, pin_memory=True,
                                           persistent_workers=False,
                                           prefetch_factor=4)
        dummy_element = next(iter(training_dataset))
    else:
        training_dataset   = MotionDataset(TRAINING_PATH)
        validation_dataset = MotionDataset(VALIDATION_PATH)
        training_dataloader   = DataLoader(training_dataset, batch_size=args.batch_size,
                                           num_workers=4, pin_memory=True,
                                           persistent_workers=True,
                                           prefetch_factor=4,
                                           multiprocessing_context='spawn')
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size,
                                           num_workers=4, pin_memory=True,
                                           persistent_workers=False,
                                           prefetch_factor=4,
                                           multiprocessing_context='spawn')
        dummy_element = training_dataset[0]

    # =====================================================================
    # Input feature shapes (derived from a dummy sample)
    # =====================================================================
    agent_input_continuous       = dummy_element['agent_input_continuous']
    static_roadgraph_input       = dummy_element['static_roadgraph_polyline_input']
    dynamic_roadgraph_continuous = dummy_element['dynamic_roadgraph_continuous']
    agent_target                 = dummy_element['agent_target']

    num_agent_continuous_features              = agent_input_continuous.size(dim=-1)
    num_static_roadgraph_features              = static_roadgraph_input.size(dim=-1)
    num_dynamic_roadgraph_continuous_features  = dynamic_roadgraph_continuous.size(dim=-1)
    num_past_timesteps                         = agent_input_continuous.size(dim=-2)
    num_future_features                        = agent_target.size(dim=-1)
    num_future_timesteps                       = agent_target.size(dim=-2)
    num_model_features                         = 256
    categorical_embedding_dim                  = 16

    # =====================================================================
    # Centroid file: compute if missing (only when the selected anchor needs it)
    # =====================================================================
    centroids = None
    if ANCHOR_TYPE in (ANCHOR_CENTROID, ANCHOR_HYBRID):
        if args.test:
            # In test mode the training_dataset isn't loaded, so we can only
            # consume an existing centroid file.
            if not os.path.exists(CENTROIDS_PATH):
                raise FileNotFoundError(
                    f"ANCHOR_TYPE={ANCHOR_TYPE!r} needs centroids at "
                    f"{CENTROIDS_PATH}, but the file does not exist. "
                    "Run training first (or compute centroids separately).")
            centroids = load_centroids(CENTROIDS_PATH)
            print(f"\nLoaded {tuple(centroids.shape)} centroids from {CENTROIDS_PATH}")
        else:
            if not os.path.exists(CENTROIDS_PATH):
                print(f"\nCentroids file not found at {CENTROIDS_PATH}.")
                print(f"Computing {NUM_CENTROIDS} centroids from training data "
                      f"(cap: {args.centroids_samples:,} endpoints)...")
                centroids = compute_and_save_centroids(
                    training_dataset, NUM_CENTROIDS, CENTROIDS_PATH,
                    max_samples=args.centroids_samples,
                )
            else:
                centroids = load_centroids(CENTROIDS_PATH)
                print(f"\nLoaded {tuple(centroids.shape)} centroids from {CENTROIDS_PATH}")
                if centroids.size(0) < NUM_MODES:
                    raise RuntimeError(
                        f"Existing centroids file has {centroids.size(0)} rows, "
                        f"but NUM_MODES={NUM_MODES} requires at least that many. "
                        f"Delete {CENTROIDS_PATH} to recompute, or lower NUM_MODES.")
    else:
        print(f"\nANCHOR_TYPE={ANCHOR_TYPE!r} — no centroids needed.")

    # =====================================================================
    # Model construction
    # =====================================================================
    model = Transformer_NN(
        num_agent_features            = num_agent_continuous_features,
        num_static_road_features      = num_static_roadgraph_features,
        num_dynamic_road_features     = num_dynamic_roadgraph_continuous_features,
        num_past_timesteps            = num_past_timesteps,
        num_model_features            = num_model_features,
        categorical_embedding_dim     = categorical_embedding_dim,
        num_future_trajectories       = NUM_MODES,
        num_future_timesteps          = num_future_timesteps,
        num_future_features           = num_future_features,
        anchor_type                   = ANCHOR_TYPE,
        centroids                     = centroids,
    )
    model.to(device)
    print(f"\nModel constructed: K={NUM_MODES} modes, anchor={ANCHOR_TYPE!r}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = NLL_Loss()
    NUM_EPOCHS = args.epochs

    # =====================================================================
    # Resume / warm-start (mutually exclusive — resume takes priority)
    # =====================================================================
    # Resume: load full bundle and continue from saved epoch. Strict shape
    #         match required for both model and optimizer state.
    # Warm-start: copy only parameters whose name and shape match; anything
    #         else keeps its fresh init (e.g. switching K, switching anchor
    #         type, adding new modules).
    start_epoch    = 0
    best_val_loss  = float('inf')
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            opt_state = ckpt.get('optimizer_state_dict')
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
                opt_msg = "with saved optimizer state"
            else:
                opt_msg = "optimizer reset (no saved state)"
            start_epoch    = ckpt.get('epoch', 0)
            best_val_loss  = ckpt.get('best_val_loss', float('inf'))
            print(f"  Restored model {opt_msg}. Continuing from epoch {start_epoch+1}, "
                  f"best_val_loss={best_val_loss:.4f}")
        else:
            model.load_state_dict(ckpt)
            print("  Loaded weights only (legacy format). "
                  "Optimizer state reset; epoch counter starts at 1.")
    elif args.warm_start:
        print(f"\nWarm-starting from: {args.warm_start}")
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        src_state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        dst_state = model.state_dict()
        loaded, skipped_missing, skipped_shape = [], [], []
        for name, dst_param in dst_state.items():
            if name not in src_state:
                skipped_missing.append(name)
                continue
            src_param = src_state[name]
            if src_param.shape != dst_param.shape:
                skipped_shape.append(
                    f"{name} (src {tuple(src_param.shape)} != dst {tuple(dst_param.shape)})")
                continue
            dst_state[name].copy_(src_param)
            loaded.append(name)
        model.load_state_dict(dst_state)
        print(f"  Loaded {len(loaded)} / {len(dst_state)} parameter tensors.")
        if skipped_shape:
            print(f"  Skipped {len(skipped_shape)} due to shape mismatch (kept random init):")
            for s in skipped_shape:
                print(f"    - {s}")
        if skipped_missing:
            print(f"  Skipped {len(skipped_missing)} not present in source (kept random init):")
            for s in skipped_missing[:10]:
                print(f"    - {s}")
            if len(skipped_missing) > 10:
                print(f"    ... and {len(skipped_missing)-10} more")
        print("  Epoch counter starts at 1, optimizer fresh.")

    # =====================================================================
    # Training loop
    # =====================================================================
    if not args.test:
        print(f"\nTraining with {args.num_train_files}/1000 files per epoch, batch size {args.batch_size}")
        print("\nStarting training...")
        for epoch in range(start_epoch, NUM_EPOCHS):
            # Shuffle and subsample the training files for this epoch so we
            # cover the full dataset over multiple epochs without ever
            # touching every file in a single epoch.
            training_dataset.set_epoch(epoch, num_files=args.num_train_files)

            model.train()
            train_loss    = 0.0
            train_batches = 0
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"{'='*50}")

            for batch_idx, dataset_element in enumerate(training_dataloader):
                agents_cont       = dataset_element['agent_input_continuous'].to(device)
                agents_cat        = dataset_element['agent_input_categorical'].to(device)
                agents_valid      = dataset_element['agent_input_valid'].to(device)
                static_road       = dataset_element['static_roadgraph_polyline_input'].to(device)
                static_road_valid = dataset_element['static_roadgraph_polyline_valid'].to(device)
                dyn_road_cont     = dataset_element['dynamic_roadgraph_continuous'].to(device)
                dyn_road_cat      = dataset_element['dynamic_roadgraph_categorical'].to(device)
                dyn_road_valid    = dataset_element['dynamic_roadgraph_valid'].to(device)
                agent_target       = dataset_element['agent_target'].to(device)
                agent_target_valid = dataset_element['agent_target_valid'].to(device)
                tracks_to_predict  = dataset_element['tracks_to_predict'].to(device)

                optimizer.zero_grad()
                trajectories, probs = model(
                    agents_cont, agents_cat, agents_valid,
                    static_road, static_road_valid,
                    dyn_road_cont, dyn_road_cat, dyn_road_valid,
                )
                loss = loss_fn(trajectories, probs, agent_target,
                               agent_target_valid, tracks_to_predict)
                loss.backward()
                # Per-step grad clip caps any single batch's update; doesn't
                # protect against compounded optimizer-state drift but
                # filters out single-batch spikes.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss    += loss.item()
                train_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    print(f"Batch {batch_idx+1}, "
                          f"Done with {batch_idx*args.batch_size+args.batch_size} samples  "
                          f"Loss: {loss.item():.4f}", flush=True)

            avg_train_loss = train_loss / train_batches

            # Per-epoch checkpoint: save before validation so weights are
            # never lost. Bundle is full state (model + optimizer + counter +
            # best_val) so --resume can pick up seamlessly.
            now = datetime.datetime.now()
            filename = f"transformer_model_epoch_{epoch+1}_{now.strftime('%Y%m%d_%H%M%S')}.pt"
            path = os.path.join(SAVE_DIR, filename)
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss':        best_val_loss,
            }, path)
            print(f"Checkpoint saved: {path}")

            # ---- Validation ----
            model.eval()
            val_loss    = 0.0
            val_batches = 0
            with torch.no_grad():
                for dataset_element in validation_dataloader:
                    agents_cont       = dataset_element['agent_input_continuous'].to(device)
                    agents_cat        = dataset_element['agent_input_categorical'].to(device)
                    agents_valid      = dataset_element['agent_input_valid'].to(device)
                    static_road       = dataset_element['static_roadgraph_polyline_input'].to(device)
                    static_road_valid = dataset_element['static_roadgraph_polyline_valid'].to(device)
                    dyn_road_cont     = dataset_element['dynamic_roadgraph_continuous'].to(device)
                    dyn_road_cat      = dataset_element['dynamic_roadgraph_categorical'].to(device)
                    dyn_road_valid    = dataset_element['dynamic_roadgraph_valid'].to(device)
                    agent_target       = dataset_element['agent_target'].to(device)
                    agent_target_valid = dataset_element['agent_target_valid'].to(device)
                    tracks_to_predict  = dataset_element['tracks_to_predict'].to(device)

                    trajectories, probs = model(
                        agents_cont, agents_cat, agents_valid,
                        static_road, static_road_valid,
                        dyn_road_cont, dyn_road_cat, dyn_road_valid,
                    )
                    loss = loss_fn(trajectories, probs, agent_target,
                                   agent_target_valid, tracks_to_predict)
                    val_loss    += loss.item()
                    val_batches += 1

                    if val_batches % 20 == 0:
                        print(f"  [val] Batch {val_batches}, running avg loss: "
                              f"{val_loss/val_batches:.4f}", flush=True)

            avg_val_loss = val_loss / val_batches

            print(f'\n{"="*50}')
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss:   {avg_val_loss:.4f}')
            print(f'{"="*50}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(SAVE_DIR, "best_model.pt")
                torch.save({
                    'epoch':                epoch + 1,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss':        best_val_loss,
                }, best_path)
                print(f"New best model saved: {best_path}")

        print("\n✓ Training complete!")

    # =====================================================================
    # Test mode: load checkpoint, run forward, save a visualization PNG
    # =====================================================================
    else:
        if args.model_path is None:
            raise ValueError("Must specify --model-path for testing mode")

        print(f"\nLoading model from: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            for dataset_element in test_dataloader:
                agents_cont       = dataset_element['agent_input_continuous'].to(device)
                agents_cat        = dataset_element['agent_input_categorical'].to(device)
                agents_valid      = dataset_element['agent_input_valid'].to(device)
                static_road       = dataset_element['static_roadgraph_polyline_input'].to(device)
                static_road_valid = dataset_element['static_roadgraph_polyline_valid'].to(device)
                dyn_road_cont     = dataset_element['dynamic_roadgraph_continuous'].to(device)
                dyn_road_cat      = dataset_element['dynamic_roadgraph_categorical'].to(device)
                dyn_road_valid    = dataset_element['dynamic_roadgraph_valid'].to(device)
                agent_target       = dataset_element['agent_target'].to(device)
                agent_target_valid = dataset_element['agent_target_valid'].to(device)
                tracks_to_predict  = dataset_element['tracks_to_predict'].to(device)

                trajectories, probs = model(
                    agents_cont, agents_cat, agents_valid,
                    static_road, static_road_valid,
                    dyn_road_cont, dyn_road_cat, dyn_road_valid,
                )

                loss = loss_fn(trajectories, probs, agent_target,
                               agent_target_valid, tracks_to_predict)
                print(f"Loss: {loss.item():.4f}")

                model_output = {
                    'agent_trajs': trajectories,
                    'agent_probs': probs,
                }
                model_input = {
                    'agent_input':                  agents_cont,
                    'agent_input_valid':            agents_valid,
                    'agent_target':                 agent_target,
                    'agent_target_valid':           agent_target_valid,
                    'static_roadgraph_input':       static_road,
                    'static_roadgraph_valid':       static_road_valid,
                    'dynamic_roadgraph_continuous': dyn_road_cont,
                    'dynamic_roadgraph_valid':      dyn_road_valid,
                    'is_sdc':                       dataset_element['is_sdc'].to(device),
                    'tracks_to_predict':            tracks_to_predict,
                }
                os.makedirs("./tmp", exist_ok=True)
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f"./tmp/test_visualization_{timestamp}.png"
                visualize_model_inputs_and_output(model_input, model_output, save_path=save_path)
