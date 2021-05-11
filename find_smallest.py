from PIL import Image
#import cv2
from os import walk

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




base_paths = [ "/home/cavtheman/skolearbejde/degridifier/data/nogrid", "/home/cavtheman/skolearbejde/degridifier/data/grid", "/home/cavtheman/skolearbejde/degridifier/data/partialgrid" ]

paths = [ get_all_files (path) for path in base_paths ]
paths = [item for sublist in paths for item in sublist]

w = float("inf")
h = float("inf")

low = 400

small_counter = 0
small_list = []
for img_path in paths:
    img = Image.open (img_path)
    this_w, this_h = img.size

    if this_w <= low or this_h <= low:
        small_counter += 1
        small_list.append(img_path)
    if this_w < w :
        w = this_w
    if this_h < h :
        h = this_h

# 234 296
print (w,h)
print (small_counter)
print (small_list)
