"""Standalone centroid pre-computation.

Mirrors the auto-compute path inside transformer_train.py but skips
everything else (model build, training loop). Useful for timing the
extraction in isolation or for re-running k-means with different K /
sample-cap values.

Defaults match the training script's DECODER ANCHOR CONFIG block.
"""
import argparse
import os
import time

from utils.data.motion_dataset import MotionDataset, PreprocessedMotionDataset
from utils.model.endpoint_anchors import compute_and_save_centroids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--preprocessed', action='store_true',
        help='Use preprocessed .pt files instead of TFRecords (much faster).')
    parser.add_argument(
        '--preprocessed-root', type=str, default='./data/preprocessed')
    parser.add_argument(
        '--local-data', action='store_true', default=True,
        help='Use ./local_data/ TFRecords (default).')
    parser.add_argument(
        '--gcsfuse', action='store_true',
        help='Use the gcsfuse mount at ./data/uncompressed/tf_example/.')
    parser.add_argument(
        '--num-centroids', type=int, default=6,
        help='K — number of cluster centers (default: 6).')
    parser.add_argument(
        '--max-samples', type=int, default=100_000,
        help='Cap on number of endpoints to collect for k-means (default: 100k).')
    parser.add_argument(
        '--output', type=str,
        default='./models/trained_weights/intention_centroids.pt',
        help='Where to save the centroid tensor.')
    args = parser.parse_args()

    if args.preprocessed:
        path = os.path.join(args.preprocessed_root, 'training')
        print(f"Dataset: preprocessed .pt at {path}")
        dataset = PreprocessedMotionDataset(path)
    elif args.gcsfuse:
        path = './data/uncompressed/tf_example/training/'
        print(f"Dataset: gcsfuse TFRecords at {path}")
        dataset = MotionDataset(path)
    else:
        path = './local_data/training/'
        print(f"Dataset: local TFRecords at {path}")
        dataset = MotionDataset(path)

    print(f"K (num centroids):       {args.num_centroids}")
    print(f"Max samples for k-means: {args.max_samples:,}")
    print(f"Output:                  {args.output}\n")

    t0 = time.time()
    compute_and_save_centroids(
        dataset, args.num_centroids, args.output, max_samples=args.max_samples,
    )
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s "
          f"({elapsed/60:.1f}min, {elapsed/3600:.2f}h)")
