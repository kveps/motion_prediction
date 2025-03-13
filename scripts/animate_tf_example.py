from IPython.core.display import HTML
import tensorflow as tf
from utils.data.features_description import get_features_description
from utils.viz.visualize_scenario import (
    visualize_all_agents_smooth,
    create_animation,
)


def display_tf_example(file_path):
    fd = get_features_description()
    if file_path == '':
        file_path = './data/uncompressed/tf_example/training/training_tfexample.tfrecord-00906-of-01000'

    dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    data = next(dataset.as_numpy_iterator())
    parsed = tf.io.parse_single_example(data, fd)

    images = visualize_all_agents_smooth(parsed)
    anim = create_animation(images)
    data = HTML(anim.to_html5_video()).data
    # write data to htm file and open in browser
    with open("tmp/data.html", "w") as file:
        print("File written to ./tmp/data.html, open it in the browser to see")
        file.write(data)


def main():
    file_path = input(
        "enter tf example full file path (empty for default): ")
    display_tf_example(file_path)


if __name__ == '__main__':
    main()
