import os
from tkinter import *
from PIL import Image,ImageTk
import numpy as np
import h5py
import cv2

class H5ImageDatabase():

    def __init__(self, num_entries, image_height, image_width, filepath):
        input_shape = (num_entries, image_height, image_width, 3)
        self.height = image_height
        self.width = image_width
        self.hdf5_file = h5py.File(filepath, mode='w')
        self.hdf5_file.create_dataset("inputs", input_shape, np.int8)
        self.hdf5_file.create_dataset("input_labels", (num_entries, num_entries, num_entries, num_entries, num_entries, num_entries, num_entries), np.int8)

class Window(Frame):

    INITIAL_HEIGHT = 768
    INITIAL_WIDTH = 1024
    IMAGE_DIR="images"
    DB_IMAGE_HEIGHT = 16*30
    DB_IMAGE_WIDTH = 9*30
    DB_FILE_NAME = "h5.db"

    def __init__(self, master=None):
        self.current_image_index = -1
        self.original = None
        self.image = None
        self.converted_image = None
        self.image_on_canvas = None
        self.comnverted_image_for_h5 = None
        self.show_cv2_image = False
        self.height = self.INITIAL_HEIGHT
        self.width = self.INITIAL_WIDTH
        for _,_,filenames in os.walk(os.getcwd() + os.sep + self.IMAGE_DIR):
            self.filenames = filenames

        self.database = H5ImageDatabase(len(self.filenames), self.DB_IMAGE_HEIGHT, self.DB_IMAGE_WIDTH, 
                                        os.getcwd()+os.sep+self.DB_FILE_NAME)
        self._reset_image_presences()

        Frame.__init__(self, master)
        self.button_bar = Label(self, text="Label TEXT")
        self.bind("<Configure>", self.resize)
        self.em_butt = Checkbutton(self.button_bar, text="Emma", variable=self.image_presences["emma"],
                                   offvalue=False, onvalue=True, anchor=W)
        self.em_butt.pack(side=LEFT)   
        self.jacob_butt = Checkbutton(self.button_bar, text="Jacob", variable=self.image_presences["jacob"],
                                   offvalue=False, onvalue=True, anchor=W)
        self.jacob_butt.pack(side=LEFT)   
        self.mim_butt = Checkbutton(self.button_bar, text="Mim", variable=self.image_presences["mim"],
                                   offvalue=False, onvalue=True, anchor=W)
        self.mim_butt.pack(side=LEFT)   
        self.ga_butt = Checkbutton(self.button_bar, text="Ga", variable=self.image_presences["ga"],
                                   offvalue=False, onvalue=True, anchor=W)
        self.ga_butt.pack(side=LEFT)   
        self.kaiser_butt = Checkbutton(self.button_bar, text="Kaiser", variable=self.image_presences["kaiser"],
                                   offvalue=False, onvalue=True, anchor=W)
        self.kaiser_butt.pack(side=LEFT)   
        self.weenie_butt = Checkbutton(self.button_bar, text="Weenie", variable=self.image_presences["weenie"],
                                   offvalue=False, onvalue=True, anchor=W)
        self.weenie_butt.pack(side=LEFT)   
        self.the_chee_butt = Checkbutton(self.button_bar, text="Archie", variable=self.image_presences["the_chee"],
                                   offvalue=False, onvalue=True, anchor=W)
        self.the_chee_butt.pack(side=LEFT)   
        self.next_butt = Button(self.button_bar, text="Next Image", command=self.next_image, anchor=W)
        self.next_butt.pack(side=LEFT)   
        self.next_and_save_butt = Button(self.button_bar, text="Save and Next Image", command=self.next_image, anchor=W)
        self.next_and_save_butt.pack(side=LEFT)   
        self.show_cv2_butt = Checkbutton(self.button_bar, text="Show Coverted Image", variable=self.show_cv2_image,
                                   offvalue=False, onvalue=True, anchor=W, command=self.toggle_image)
        self.show_cv2_butt.pack(side=LEFT)   

        self.button_bar.pack()     
        self.display = Canvas(self, bd=0, highlightthickness=0)
        self.display.pack()

        self.master = master
        self.init_window()

    def _reset_image_presences(self):
        self.image_presences = {"emma":IntVar(value=0), "ga":IntVar(value=0), "mim":IntVar(value=0), 
                                "jacob":IntVar(value=0), "kaiser":IntVar(value=0), "the_chee":IntVar(value=0), 
                                "weenie":IntVar(value=0)}

    def init_window(self):
        self.master.title("Manual Image Classifier")
        self.pack(fill=BOTH, expand=1)
        menu = Menu(self.master)
        self.master.config(menu=menu)
        file = Menu(menu)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)
        edit = Menu(menu)
        edit.add_command(label="Next Image", command=self.next_image)
        menu.add_cascade(label="Edit", menu=edit)
        self.next_image()

    def next_image(self):
        self._reset_image_presences()
        # the graphical part
        self.current_image_index += 1
        filename = os.getcwd() + os.sep + self.IMAGE_DIR + os.sep + self.filenames[self.current_image_index]
        self.original = Image.open(filename)
        self._reset_image_presences()
        # now the conversion to a format suitable for db and ml
        cv2_im = cv2.imread(filename)
        cv2_im = cv2.resize(cv2_im, (self.database.width, self.database.height), interpolation=cv2.INTER_CUBIC)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        self.converted_inage = ImageTk.PhotoImage(Image.fromarray(cv2_im))
        self.image = ImageTk.PhotoImage(self.original)
        self.image_on_canvas = self.display.create_image(0, 0, image=self._get_appropriate_image(), 
                                                         anchor=NW, tags="IMG")
        self.display.itemconfig(self.image_on_canvas, image = self._get_appropriate_image())
        self.resize(None)
        print("Next image is " + self.filenames[self.current_image_index])

    def _get_appropriate_image(self):
        if (self.show_cv2_image):
            return self.converted_inage
        return self.image

    def resize(self, event=None):
        if (event):
            self.height = event.height
            self.width = event.width
        size = (self.height, self.width)
        resized = self.original.resize(size,Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(resized)
        self.display.config(width=self.width, height=self.height)
        filename = os.getcwd() + os.sep + self.IMAGE_DIR + os.sep + self.filenames[self.current_image_index]
        cv2_im = cv2.imread(filename)
        cv2_im = cv2.resize(cv2_im, (self.database.height, self.database.width), interpolation=cv2.INTER_CUBIC)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        self.converted_image = ImageTk.PhotoImage(Image.fromarray(cv2_im))
        self.image_on_canvas = self.display.create_image(0, 0, image=self._get_appropriate_image(), 
                                                         anchor=NW, tags="IMG")
        self.display.itemconfig(self.image_on_canvas, image = self._get_appropriate_image())

    def toggle_image(self):
        self.show_cv2_image = self.show_cv2_image != True

    def client_exit(self):
        exit()

    class PersonSelector():

        def __init__(self, person, owner):
            self.person = person
            self.owner = owner

        def __call__(self):
            self.owner.image_presences[self.person] = IntVar(value=1)

root=Tk()
app = Window(root)
root.mainloop()






