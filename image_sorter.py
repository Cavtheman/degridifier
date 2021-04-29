from PIL import ImageTk, Image
import tkinter as tk
from os import walk
import os
import re

class ImageSorter ():
    def get_all_files (self, path):
        if not path[-1] == "/":
            path = path+"/"

        (_, sub_folders, filenames) = next(walk(path))
        filenames = [ path + elem for elem in filenames ]
        #print(sub_folders)
        if sub_folders:
            sub_filenames = [ self.get_all_files (path + elem) for elem in sub_folders ]
            sub_filenames = [ item for sublist in sub_filenames for item in sublist ]
            filenames.extend (sub_filenames)

        return filenames

    def partial_grid (self, e):
        filepath = self.filepaths[self.img_ind]
        filename = re.search (".*\/(.*)", filepath).group(1)
        new_path = self.base_path + "data/partialgrid/" + filename
        os.rename(filepath, new_path)
        self.go_next()

    def grid (self, e):
        filepath = self.filepaths[self.img_ind]
        filename = re.search (".*\/(.*)", filepath).group(1)
        new_path = self.base_path + "data/grid/" + filename
        os.rename(filepath, new_path)
        self.go_next()

    def no_grid (self, e):
        filepath = self.filepaths[self.img_ind]
        filename = re.search (".*\/(.*)", filepath).group(1)
        new_path = self.base_path + "data/nogrid/" + filename
        os.rename(filepath, new_path)
        self.go_next()

    def uncertain (self, e):
        filepath = self.filepaths[self.img_ind]
        filename = re.search (".*\/(.*)", filepath).group(1)
        new_path = self.base_path + "data/uncertain/" + filename
        os.rename(filepath, new_path)
        self.go_next()

    def wrong_img (self, e):
        filepath = self.filepaths[self.img_ind]
        filename = re.search (".*\/(.*)", filepath).group(1)
        new_path = self.base_path + "data/wrong/" + filename
        os.rename(filepath, new_path)
        self.go_next()

    def __init__ (self, path):
        self.base_path = "/home/cavtheman/skolearbejde/degridifier/"
        self.filepaths = self.get_all_files (path)
        self.root = tk.Tk ()
        render = ImageTk.PhotoImage (file=self.filepaths[0])
        self.label = tk.Label (image = render)
        self.label.image = render
        self.label.place (x=0,y=0)

        self.img_ind = 0
        self.num_files = len(self.filepaths)

        self.root.geometry ("2560x1440")

        self.root.bind ("<Up>", self.partial_grid)
        self.root.bind ("<Left>", self.no_grid)
        self.root.bind ("<Right>", self.grid)
        self.root.bind ("<Down>", self.uncertain)
        self.root.bind ("<BackSpace>", self.wrong_img)
        self.root.bind ("<Escape>", self.kill_window)
        self.root.bind ("<space>", self.go_next)
        self.root.mainloop ()

    def go_next (self, e=None):
        if self.img_ind < self.num_files:
            self.img_ind = self.img_ind + 1
            img2 = ImageTk.PhotoImage(Image.open(self.filepaths[self.img_ind]))
            self.label.configure(image=img2)
            self.label.image = img2
        else:
            self.kill_window()


    def kill_window (self, e=None):
        self.root.destroy()

path = "/home/cavtheman/skolearbejde/degridifier/data/"
#all_images = get_all_files(path)
test = ImageSorter(path)
first = "mn41m9-Elven_Crypt__26x36_-1kt0676p61s61.jpg"
second = "mn5gtx-The_Swamp_-_Large_gridless_map-mf1q1vj2l1s61.jpg"



#for image_path in all_images:
#
#    break
