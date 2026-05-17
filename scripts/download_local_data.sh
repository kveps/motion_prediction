#!/usr/bin/env bash
# Download a subset of Waymo Open Motion TFRecord shards to local_data/.
# Resumable: already-complete files (matching remote size) are skipped.
# Conservative parallelism: PARALLEL files per gcloud invocation,
# per-file thread slicing disabled to keep RAM bounded.

set -euo pipefail

BUCKET="gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example"
DEST_ROOT="$(cd "$(dirname "$0")/.." && pwd)/local_data"

TRAIN_COUNT="${TRAIN_COUNT:-50}"
VAL_COUNT="${VAL_COUNT:-20}"
PARALLEL="${PARALLEL:-4}"

# Total shard counts per split (used to format the "-of-XXXXX" suffix in filenames).
declare -A SPLIT_TOTAL=(
  [training]=1000
  [validation]=150
)

# Disable per-object slicing so each file is one stream, not N threads.
# Combined with batching of $PARALLEL files per gcloud call, peak concurrency = $PARALLEL.
export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0

mkdir -p "$DEST_ROOT/training" "$DEST_ROOT/validation"

shard_name() {
  # $1=split, $2=index
  local total="${SPLIT_TOTAL[$1]}"
  printf "%s_tfexample.tfrecord-%05d-of-%05d" "$1" "$2" "$total"
}

remote_size() {
  gcloud storage ls -l "$1" 2>/dev/null | awk 'NR==1 {print $1}'
}

needs_download() {
  local remote="$1" local_path="$2"
  if [[ ! -f "$local_path" ]]; then
    echo "$remote"; return
  fi
  local r_size l_size
  r_size="$(remote_size "$remote")"
  l_size="$(stat -c%s "$local_path")"
  if [[ "$r_size" != "$l_size" ]]; then
    rm -f "$local_path"
    echo "$remote"
  fi
}

download_split() {
  local split="$1" count="$2"
  local dest="$DEST_ROOT/$split"
  local to_fetch=()

  echo ">>> Checking $split ($count shards)..."
  for ((i=0; i<count; i++)); do
    local name remote local_path missing
    name="$(shard_name "$split" "$i")"
    remote="$BUCKET/$split/$name"
    local_path="$dest/$name"
    missing="$(needs_download "$remote" "$local_path")"
    [[ -n "$missing" ]] && to_fetch+=("$missing")
  done

  local total="${#to_fetch[@]}"
  if [[ "$total" -eq 0 ]]; then
    echo "    all $count shards already present, nothing to do."
    return
  fi
  echo "    $total shard(s) to fetch (of $count requested), $PARALLEL at a time."

  local idx=0
  while ((idx < total)); do
    local end=$((idx + PARALLEL))
    ((end > total)) && end=$total
    local batch=("${to_fetch[@]:idx:end-idx}")
    echo "    [$((idx+1))-$end / $total] downloading..."
    gcloud storage cp "${batch[@]}" "$dest/"
    idx=$end
  done
  echo "    $split done."
}

echo "Destination: $DEST_ROOT"
echo "Parallel:    $PARALLEL"
echo "Train/Val:   $TRAIN_COUNT / $VAL_COUNT"
echo

download_split training "$TRAIN_COUNT"
download_split validation "$VAL_COUNT"

echo
echo "All done. Local sizes:"
du -sh "$DEST_ROOT/training" "$DEST_ROOT/validation"
