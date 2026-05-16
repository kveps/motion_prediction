import glob
import math
import os
import random
import tensorflow as tf
import torch
from torch.utils.data import Dataset, IterableDataset
from utils.data.features_description import get_features_description
from utils.viz.visualize_scenario import (
    visualize_polylines,
    visualize_scenario_image,
)
from utils.data.data_processing_helpers import (
    filter_roadgraph_by_proximity,
    get_data_file_names,
    transform_parsed_dataset_to_av_frame,
    arrange_agent_model_input,
    arrange_agent_model_target,
    arrange_dynamic_roadgraph_model_input,
    arrange_static_roadgraph_model_input,
    arrange_static_roadgraph_polyline_model_input,
)


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    fd = get_features_description()
    parsed = tf.io.parse_single_example(example_proto, fd)
    # Translate the data points around the AV center i.e. AV is at origin
    transformed = transform_parsed_dataset_to_av_frame(parsed)
    # Keep only roadgraph points within 80 m of the AV
    filtered = filter_roadgraph_by_proximity(transformed)
    return filtered

class MotionDataset(IterableDataset):
    def __init__(self, data_path, shuffle=False):
        self.data_path = data_path
        self.shuffle = shuffle

        # Handle both local and GCS paths
        if data_path.startswith('gs://'):
            path = data_path.rstrip('/')
            data_files = tf.io.gfile.glob(path + '/*.tfrecord-*')
            if not data_files:
                raise ValueError(f"No .tfrecord-* files found at {path}")
            print(f"Found {len(data_files)} files in GCS")
        else:
            data_files = get_data_file_names(data_path)
            data_files = [data_path + file for file in data_files]

        self.data_files = data_files

        # Full pipeline for __len__ / __getitem__ / get_tf_dataset
        tf_dataset = tf.data.TFRecordDataset(data_files)
        self.parsed_tf_dataset = tf_dataset.map(
            _parse_function, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        if shuffle:
            self.parsed_tf_dataset = self.parsed_tf_dataset.shuffle(
                buffer_size=1000)

    def set_epoch(self, epoch, num_files=None):
        rng = random.Random(epoch)
        files = list(self.data_files)
        rng.shuffle(files)
        self._epoch_files = files[:num_files] if num_files is not None else files

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['parsed_tf_dataset']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        tf_dataset = tf.data.TFRecordDataset(self.data_files)
        self.parsed_tf_dataset = tf_dataset.map(
            _parse_function, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        if self.shuffle:
            self.parsed_tf_dataset = self.parsed_tf_dataset.shuffle(buffer_size=1000)

    def __len__(self):
        count = 0
        for element in self.parsed_tf_dataset:
            count += 1
        return count

    def __iter__(self):
        epoch_files = getattr(self, '_epoch_files', self.data_files)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = epoch_files
        else:
            per_worker = math.ceil(len(epoch_files) / worker_info.num_workers)
            start = worker_info.id * per_worker
            files = epoch_files[start:start + per_worker]

        # Each DataLoader worker gets its own private TF thread pool to prevent
        # contention across workers on the shared global pool.
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        options.threading.max_intra_op_parallelism = 1

        tf_dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
        parsed = tf_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle:
            parsed = parsed.shuffle(buffer_size=1000)
        parsed = parsed.prefetch(tf.data.AUTOTUNE).with_options(options)

        for element in parsed:
            # Convert TensorFlow tensors to NumPy arrays
            numpy_element = {key: value.numpy() for key, value in element.items()}

            # Convert NumPy arrays to PyTorch tensors
            torch_element = {key: torch.tensor(value)
                            for key, value in numpy_element.items()}

            # Static roadgraph
            static_roadgraph_polyline_input, static_roadgraph_polyline_valid = arrange_static_roadgraph_polyline_model_input(
                torch_element)
            # Dynamic roadgraph (separate continuous and categorical)
            dynamic_roadgraph_cont, dynamic_roadgraph_cat, dynamic_roadgraph_valid = arrange_dynamic_roadgraph_model_input(
                torch_element)
            # Agent states (separate continuous and categorical)
            agent_input_cont, agent_input_cat, agent_input_valid = arrange_agent_model_input(
                torch_element)
            # Agent targets
            agent_target, agent_target_valid = arrange_agent_model_target(
                torch_element)
            # Is AV/SDC
            is_sdc = torch_element['state/is_sdc']
            # Tracks to predict
            tracks_to_predict = torch_element['state/tracks_to_predict']

            yield {
                'static_roadgraph_polyline_input': static_roadgraph_polyline_input,
                'static_roadgraph_polyline_valid': static_roadgraph_polyline_valid,
                'dynamic_roadgraph_continuous': dynamic_roadgraph_cont,
                'dynamic_roadgraph_categorical': dynamic_roadgraph_cat,
                'dynamic_roadgraph_valid': dynamic_roadgraph_valid,
                'agent_input_continuous': agent_input_cont,
                'agent_input_categorical': agent_input_cat,
                'agent_input_valid': agent_input_valid,
                'agent_target': agent_target,
                'agent_target_valid': agent_target_valid,
                'is_sdc': is_sdc,
                'tracks_to_predict': tracks_to_predict,
            }

    def __getitem__(self, idx):
        # Take the element at the given index
        element = list(self.parsed_tf_dataset.skip(idx).take(1))[0]

        # Convert TensorFlow tensors to NumPy arrays
        numpy_element = {key: value.numpy() for key, value in element.items()}

        # Convert NumPy arrays to PyTorch tensors
        torch_element = {key: torch.tensor(value)
                         for key, value in numpy_element.items()}

        # Static roadgraph
        static_roadgraph_polyline_input, static_roadgraph_polyline_valid = arrange_static_roadgraph_polyline_model_input(
            torch_element)
        # Dynamic roadgraph (separate continuous and categorical)
        dynamic_roadgraph_cont, dynamic_roadgraph_cat, dynamic_roadgraph_valid = arrange_dynamic_roadgraph_model_input(
            torch_element)
        # Agent states (separate continuous and categorical)
        agent_input_cont, agent_input_cat, agent_input_valid = arrange_agent_model_input(
            torch_element)
        # Agent targets
        agent_target, agent_target_valid = arrange_agent_model_target(
            torch_element)
        # Is AV/SDC
        is_sdc = torch_element['state/is_sdc']
        # Tracks to predict
        tracks_to_predict = torch_element['state/tracks_to_predict']

        return {
            'static_roadgraph_polyline_input': static_roadgraph_polyline_input,
            'static_roadgraph_polyline_valid': static_roadgraph_polyline_valid,
            'dynamic_roadgraph_continuous': dynamic_roadgraph_cont,
            'dynamic_roadgraph_categorical': dynamic_roadgraph_cat,
            'dynamic_roadgraph_valid': dynamic_roadgraph_valid,
            'agent_input_continuous': agent_input_cont,
            'agent_input_categorical': agent_input_cat,
            'agent_input_valid': agent_input_valid,
            'agent_target': agent_target,
            'agent_target_valid': agent_target_valid,
            'is_sdc': is_sdc,
            'tracks_to_predict': tracks_to_predict,
        }

    def get_full_torch_element(self, idx):
        # Take the element at the given index
        element = list(self.parsed_tf_dataset.skip(idx).take(1))[0]

        # Convert TensorFlow tensors to NumPy arrays
        numpy_element = {key: value.numpy() for key, value in element.items()}

        # Convert NumPy arrays to PyTorch tensors
        torch_element = {key: torch.tensor(value)
                         for key, value in numpy_element.items()}

        return torch_element

    def get_tf_dataset(self):
        return self.parsed_tf_dataset


class PreprocessedMotionDataset(IterableDataset):
    """Fast dataset that reads pre-saved .pt files instead of TFRecords.

    Each .pt file is a list of sample dicts produced by preprocess_dataset.py.
    Workers split files evenly so there is no redundant I/O.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = sorted(glob.glob(os.path.join(data_path, '*.pt')))
        if not self.data_files:
            raise ValueError(f"No .pt files found in {data_path}")
        self._epoch_files = self.data_files

    def set_epoch(self, epoch, num_files=None):
        rng = random.Random(epoch)
        files = list(self.data_files)
        rng.shuffle(files)
        self._epoch_files = files[:num_files] if num_files is not None else files

    def __iter__(self):
        epoch_files = getattr(self, '_epoch_files', self.data_files)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = epoch_files
        else:
            per_worker = math.ceil(len(epoch_files) / worker_info.num_workers)
            start = worker_info.id * per_worker
            files = epoch_files[start:start + per_worker]

        for path in files:
            for sample in torch.load(path, weights_only=True):
                yield sample


# Example usage
test_usage = False
if test_usage:
    directory_path = "./data/uncompressed/tf_example/training/"
    motion_dataset = MotionDataset(directory_path)
    map_polyline, validity = arrange_static_roadgraph_polyline_model_input(
        motion_dataset.get_full_torch_element(20))
    visualize_polylines(map_polyline, validity)
