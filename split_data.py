from PIL import Image
#import cv2
import os
from os import walk
import re

def get_all_files (path):
    if not path[-1] == "/":
        path = path+"/"

    (_, sub_folders, filenames) = next(walk(path))
    filenames = [ path + elem for elem in filenames ]
    #print(sub_folders)
    if sub_folders:
        sub_filenames = [ get_all_files (path + elem) for elem in sub_folders ]
        sub_filenames = [ item for sublist in sub_filenames for item in sublist ]
        filenames.extend (sub_filenames)

    return filenames


#grid_files = get_all_files ("data/grid/")
def train_val_test_split (filenames, new_loc):
    split_point = int (len(filenames) * 0.8)

    train_data = filenames[:split_point]
    val_test_data = filenames[split_point:]

    val_test_split_point = int (len(val_test_data)/2)
    val_data = val_test_data[:val_test_split_point]
    test_data = val_test_data[val_test_split_point:]
    print(len(filenames), len(train_data), len(val_data), len(test_data))
    for train, val, test in zip (train_data, val_data, test_data):
        train = re.search (".*\/(.*)", train).group(1)
        val = re.search (".*\/(.*)", val).group(1)
        test = re.search (".*\/(.*)", test).group(1)

        os.rename(new_loc[0] + new_loc[1] + train, new_loc[0] + "train/" + new_loc[1] + train)
        os.rename(new_loc[0] + new_loc[1] + val, new_loc[0] + "val/" + new_loc[1] + val)
        os.rename(new_loc[0] + new_loc[1] + test, new_loc[0] + "test/" + new_loc[1] + test)

    return train_data, val_data, test_data



train_val_test_split (get_all_files("data/grid"), ("data/", "grid/"))
train_val_test_split (get_all_files("data/nogrid"), ("data/", "nogrid/"))
