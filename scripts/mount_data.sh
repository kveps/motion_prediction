#! /usr/bin/bash

# Follow steps here if gcsfuse is not yet installed: https://cloud.google.com/storage/docs/gcsfuse-quickstart-mount-bucket

# After the first installation, just run this script

# make directory in the project repo if not already available
mkdir -p ./data

# gcloud auth
gcloud auth application-default login

# mount the waymo dataset
gcsfuse --implicit-dirs waymo_open_dataset_motion_v_1_2_1 ./data
