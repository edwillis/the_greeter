import os
from tkinter import *
from PIL import Image,ImageTk
import numpy as np
import h5py
import cv2
import configparser

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
    DB_FILE_NAME = "h5.db"

    def __init__(self, entities, image_dir, db_image_height, db_image_width, master=None):
        self.current_image_index = -1
        self.original = None
        self.image = None
        self.converted_image = None
        self.image_on_canvas = None
        self.comnverted_image_for_h5 = None
        self.show_cv2_image = False
        self.height = self.INITIAL_HEIGHT
        self.width = self.INITIAL_WIDTH
        self.image_dir="images"
        self.db_image_height = 12*45
        self.db_image_width = 16*45
        self.entities = entities
        self.entityCheckBoxes = {}
        self.image_presences = {}
        self.filenames=[]
        for _,_,filenames in os.walk(os.getcwd() + os.sep + self.image_dir):
            self.filenames = filenames

        self.database = H5ImageDatabase(len(self.filenames), self.db_image_height, self.db_image_width, 
                                        os.getcwd()+os.sep+self.DB_FILE_NAME)
        Frame.__init__(self, master)
        self.button_bar = Label(self, text="Label TEXT")
        self.bind("<Configure>", self.resize)

        for entity in self.entities:
            self.image_presences[entity] = IntVar(value=0)
        for entity in self.entities:
            self.entityCheckBoxes[entity] = Checkbutton(self.button_bar, text=entity, variable=self.image_presences[entity],
                                   offvalue=False, onvalue=True, anchor=W)
            self.entityCheckBoxes[entity].pack(side=LEFT)   
        self._reset_image_presences()        
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
        self.image_presences = {}
        for entity in self.entities:
            self.image_presences[entity] = IntVar(value=0)
            self.entityCheckBoxes[entity].deselect()

    def _get_appropriate_image(self):
        if (self.show_cv2_image):
            return self.converted_inage
        return self.image

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
        self.current_image_index += 1
        filename = os.getcwd() + os.sep + self.image_dir + os.sep + self.filenames[self.current_image_index]
        self.original = Image.open(filename)
        self._reset_image_presences()
        cv2_im = cv2.imread(filename)
        cv2_im = cv2.resize(cv2_im, (self.database.width, self.database.height), interpolation=cv2.INTER_CUBIC)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        self.converted_inage = ImageTk.PhotoImage(Image.fromarray(cv2_im))
        self.resize(None)
        print("Next image is " + self.filenames[self.current_image_index])

    def resize(self, event=None):
        if (event):
            self.height = event.height
            self.width = event.width
        size = (self.width, self.height)
        resized = self.original.resize(size,Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(resized)
        self.display.config(width=self.width, height=self.height)
        self.image_on_canvas = self.display.create_image(0, 0, image=self._get_appropriate_image(), 
                                                         anchor=NW, tags="IMG")
        self.display.itemconfig(self.image_on_canvas, image = self._get_appropriate_image())

    def toggle_image(self):
        self.show_cv2_image = self.show_cv2_image != True
        self.resize(None)

    def client_exit(self):
        exit()

    class PersonSelector():

        def __init__(self, person, owner):
            self.person = person
            self.owner = owner

        def __call__(self):
            self.owner.image_presences[self.person] = IntVar(value=1)

root=Tk()

config = configparser.ConfigParser()
config.read('scripts/config.ini')
print(config.sections())
db_image_height = config.get('database', 'imageHeight')
db_image_width = config.get('database', 'imageWidth')
image_dir = config.get('database', 'image_dir')
entities = config.get('general', 'enetitiesToLookFor').split(',')
app = Window(entities, image_dir, db_image_height, db_image_width, root)
root.mainloop()






