from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from  tensorflow.contrib.data import Iterator

beginTime = time.time()
batch_size = 100
learning_rate = 0.005
max_steps = 2
NUM_CLASSES = 200

def input_parser(img_path, label):
    one_hot = tf.one_hot(label, NUM_CLASSES)

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file, channels=3)

    return img_decoded, one_hot

LABEL_PATH = "./data/tiny-imagenet-200/words.txt"
train_dir = "./data/tiny-imagenet-200/train"
test_dir = "./data/tiny-imagenet-200/test"

# Create labels for training
wnid_labels_dict = dict()
labels_set = set()
#labels_list = []
with open(LABEL_PATH, 'r') as file:
    print("open file!")
    for line in file:
        #print(line)
        data = line.split("\t")
        for category in data[1].rstrip().split(","):
            labels_set.add(category)
        wnid_labels_dict.update({data[0]:list(data[1].rstrip().split(","))})

        # Add labels to a set to determine size of y
        # labels_list.append(data[1][:-1].split(", "))
        #labels_set = set.union(labels_set, data[1][:-1].split(", "))
        #print(labels_set)
    file.close()
labels_set = list(set(labels_set))
labels_dict = dict()
for i in range(len(labels_set)):
    temp = [False] * len(labels_set)
    temp[i] = True
    labels_dict.update({labels_set[i]: temp})

wnid_one_hot_labels = dict()
for wnid in wnid_labels_dict:
    temp_labels_list = wnid_labels_dict[wnid]
    temp_encoding = [0] * len(labels_set)
    for label in temp_labels_list:
        temp_encoding[labels_set.index(label)] = 1
    wnid_one_hot_labels.update({wnid: temp_encoding})






# Import images for training
train_file_list = []
train_label_list = []
print(len(os.listdir(train_dir)))
for folder in os.listdir(train_dir):
    for img in os.listdir(os.path.join(train_dir, str(folder + "/images"))):
        train_label_list.append(wnid_labels_dict[folder])
        train_file_list.append(os.path.join(train_dir, str(folder + "/images"), img))

test_file_list = []
test_label_list = []

for img in os.listdir(str(test_dir + "/images")):
    test_label_list.append(wnid_labels_dict[folder])
    test_file_list.append(os.path.join(train_dir, str(folder + "/images"), img))


train_images_tensor = ops.convert_to_tensor(train_file_list, dtype=dtypes.string)
train_labels_tensor = ops.convert_to_tensor(train_label_list, dtype=dtypes.int64)
test_images_tensor = ops.convert_to_tensor(test_file_list, dtype=dtypes.string)
test_labels_tensor = ops.convert_to_tensor(test_label_list, dtype=dtypes.int64)

train_data = tf.data.Dataset.from_tensor_slices((train_images_tensor, train_labels_tensor))
test_data = tf.data.Dataset.from_tensor_slices((test_images_tensor, test_labels_tensor))

train_data, train_labels = train_data.map(input_parser)
test_data, test_labels = test_data.map(input_parser)

image_size = 26*26*3
images_placeholder = tf.placeholder(tf.float32, shape=[None, image_size])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

weights =tf.Variable(tf.zeros([image_size, NUM_CLASSES]))
biases = tf.Variable(tf.zeros([NUM_CLASSES]))

logits = tf.matmul(images_placeholder, weights) + biases

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits= logits, labels= labels_placeholder))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

iterator = Iterator.from_structure(train_data.output_types,
                                   train_data.output_shapes)

next_element = iterator.get_next()

train_init_op = iterator.make_initializere(train_data)
test_init_op = iterator.make_initializere(test_data)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op)
    for i in range(10):
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training data")
            break
