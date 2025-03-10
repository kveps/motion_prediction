import sys
sys.path.append('../')

from kv_transformer.utils.viz import animate_scenario
import tensorflow as tf
from IPython.core.display import display, HTML
from kv_transformer.utils.data import features_description

def display_tf_example(file_path):
    fd = features_description.get_features_description()
    if file_path == '':
        file_path = './data/uncompressed/tf_example/training/training_tfexample.tfrecord-00906-of-01000'

    dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    data = next(dataset.as_numpy_iterator())
    parsed = tf.io.parse_single_example(data, fd)

    images = animate_scenario.visualize_all_agents_smooth(parsed)
    anim = animate_scenario.create_animation(images)
    data = HTML(anim.to_html5_video()).data
    # write data to htm file and open in browser
    with open("tmp/data.html", "w") as file:
        print("File written to ./tmp/data.html, open it in the browser to see")
        file.write(data)

def main():
    file_path = input("enter tf example full file path (just hit enter if you want to use a default example): ")
    display_tf_example(file_path)

if __name__ == '__main__':
    main()