"""The main method

"""

import os
import numpy as np
from matplotlib import image as mpimg
from Perceptron import Perceptron
import tensorflow as tf

def main():
    """Main function that creates the input and trains the perceptron
    """
    # Config
    LABEL_PATH = "./data/tiny-imagenet-200/words.txt"
    IMG_SHAPE = [64, 64, 3]
    NUM_CLASSES = 200

    # Create labels for training
    labels_dict = dict()
    #labels_set = set()
    #labels_list = []
    with open(LABEL_PATH, 'r') as file:
        print("open file!")
        for line in file:
            #print(line)
            data = line.split("\t")
            # print(data)
            labels_dict.update({data[0]:list(data[1].rstrip().split(","))})

            # Add labels to a set to determine size of y
            # labels_list.append(data[1][:-1].split(", "))
            #labels_set = set.union(labels_set, data[1][:-1].split(", "))
            #print(labels_set)
    file.close()
    # labels_set = set(labels_list)
    print("Created Labels")
    # for i in labels_dict:
    #     print(i, labels_dict[i])
    #print(labels_set)


    # Import images for training
    list_of_training_images = []
    labels = []
    img_dir = "./data/tiny-imagenet-200/train"
    for folder in os.listdir(img_dir):
        for img in os.listdir(os.path.join(img_dir, str(folder + "/images"))):
            labels.append(labels_dict[folder])
            img = os.path.join(img_dir, str(folder + "/images"), img)
            # img = os.path.join(img,)
            if not img.endswith(".JPEG"):
                continue
            arr_img = mpimg.imread(img)
            if arr_img is None:
                print("Unable to read image", img)
                continue
            list_of_training_images.append(arr_img.flatten())
    train_data = np.array(list_of_training_images)
    label_data = np.array(labels)

    print("Imported Images")

    assert train_data.shape[0] == label_data.shape[0]

    features_placeholder = tf.placeholder(train_data.dtype, train_data.shape)
    labels_placeholder = tf.placeholder(label_data.dtype, label_data.shape)

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    #dataset = dataset.map()
    batched = dataset.batch(100)
    #iterator = batched.make_one_shot_iterator()
    #inputs, labels = iterator.get_next()



    x = tf.placeholder(tf.float32, [None, 12288])
    W = tf.Variable(tf.zeros([12288, 200]))
    b = tf.Variable(tf.zeros([200]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.string, [None, 200])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # print(dataset.output_shapes)
    # print(dataset.output_types)
    session.run(train_step, feed_dict={x: dataset.output_shapes, y_: dataset.output_types})

    #eval_results = session.ev
    # sess.run()

    print(len(train_data[0]))
    print(label_data[0])

    #ptron = Perceptron(train_data, label_data, 200, mode="train")


def parse_img():
    pass


if __name__ == '__main__':
    main()
