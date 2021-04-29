import os
from os import walk
import re

def get_all_files (path, rename=True):
        if not path[-1] == "/":
            path = path+"/"

        (_, sub_folders, filenames) = next(walk(path))
        filenames = [ path + elem for elem in filenames ]

        if sub_folders:
            sub_filenames = [ get_all_files (path + elem, rename=False) for elem in sub_folders ]
            sub_filenames = [ item for sublist in sub_filenames for item in sublist ]
            filenames.extend (sub_filenames)

        new_filenames = [ re.search ("(.+\.jpg|\.png|\.mp4|\.gif)", elem).group(1) for elem in filenames ]
        return new_filenames
        for old, new in zip (filenames, new_filenames):
            os.rename (old, new)

print (get_all_files("/home/cavtheman/skolearbejde/degridifier/data/"))
