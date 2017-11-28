"""The main method

"""

import os
import numpy as np
from matplotlib import image as mpimg
#from Perceptron import Perceptron

def main():
    """Main function that creates the input and trains the perceptron
    """

    # Create labels for training
    labels_dict = dict()
    labels_set = set()
    with open("./data/tiny-imagenet-200/words.txt", 'r') as file:
        for line in file:
            #print(line)
            data = line.split("\t")
            # print(data)
            labels_dict.update({data[0]:data[1][:-1].split(",")})

            # Add labels to a set to determine size of y
            labels_set = set.union(labels_set, data[1][:-1].split(", "))
            #print(labels_set)
    file.close()
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
    print(train_data[0])
    print(label_data[0])

    #ptron = Perceptron(train_data, label_data, labels_set, mode="train")



if __name__ == '__main__':
    main()
