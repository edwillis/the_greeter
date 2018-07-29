import os
from tkinter import *
from PIL import Image,ImageTk
from pprint import pprint
import numpy as np
import h5py
import cv2
import configparser

class H5ImageDatabase():

    def __init__(self, filename):
        self.filename = filename

    def save(self, datasets):
        #         datasets = {'training': [self.image_records, len(self.image_records)}
        self.hdf5_file = h5py.File(self.filename, mode='w')
        # todo what if dataset is empty
        for dataset in datasets:
            grp = self.hdf5_file.create_group(dataset)
            fname_recs = list(datasets[dataset][0].keys())[:datasets[dataset][1]]
            fname_recs = [r.encode("ascii", "ignore") for r in fname_recs]
            # persist the image filenames
            grp.create_dataset("filenames", data=fname_recs, dtype=h5py.special_dtype(vlen=str))
            # persist the images themselves (ML inputs)
            recs = list(datasets[dataset][0].values())[:datasets[dataset][1]]
            img_recs = [ i.cv2_image for i in recs]
            img_rec_shape = (len(img_recs), img_recs[0].shape[0], img_recs[0].shape[1], img_recs[0].shape[2])
            img_dataset = grp.create_dataset("inputs", img_rec_shape, np.uint8)
            for i in range(len(img_recs)):
                img_dataset[i, ...] = img_recs[i][None]
            # persist the classifications (ML outputs)
            class_recs = [ i.classifications for i in recs]
            class_rec_shape = (len(class_recs), len(class_recs[0]))
            cls_dataset = grp.create_dataset("outputs", class_rec_shape, np.uint8)
            for i in range(len(class_recs)):
                classes = list(class_recs[i].values())
                cls_dataset[i] = classes
        self.hdf5_file.close()

    def load(self, entities):
        #         datasets = {'training': [self.image_records, len(self.image_records)}
        try:
            combined_datasets = {}
            self.hdf5_file = h5py.File(self.filename, mode='r')
            for group in self.hdf5_file:
                combined_datasets[group] = [dict(), 0]
                zips = zip(self.hdf5_file[group]["filenames"],
                           self.hdf5_file[group]["inputs"].value,
                           self.hdf5_file[group]["outputs"].value)
                for fname, input, output in zips:
                    combined_datasets[group][0][fname] = ImageRecord()
                    combined_datasets[group][0][fname].cv2_image = input
                    cls_pairs = zip(entities, output)
                    for c in cls_pairs:
                        combined_datasets[group][0][fname].classifications[c[0]] = c[1]
            self.hdf5_file.close()
            return [ combined_datasets, len(combined_datasets) ]
        except Exception as ex:
            print(ex)
            return None

class ImageRecord():

    def __init__(self):
        self.classifications = {}
        self.cv2_image = None

    def __repr__(self):
        if (self.cv2_image is not None and self.classifications):
            return "image shape:  " + str(self.cv2_image.shape) + str(self.classifications)
        return "Uninitialized"

class Window(Frame):

    INITIAL_HEIGHT = 768
    INITIAL_WIDTH = 1024
    DB_FILE_NAME = "h5.db"

    def __init__(self, entities, image_dir, db_image_height, db_image_width, master=None):
        self.current_image_index = -1
        self.image = None
        self.converted_image = None
        self.image_on_canvas = None
        self.height = self.INITIAL_HEIGHT
        self.width = self.INITIAL_WIDTH
        self.image_dir="images"
        self.db_image_height = 12*45
        self.db_image_width = 16*45
        self.entities = entities
        self.entityCheckBoxes = {}
        self.image_presences = {}
        self.database = None
        self.status_text=StringVar()
        self.image_records = {}
        self.filenames=[]
        self.current_filename=None
        for dir,_,filenames in os.walk(os.getcwd() + os.sep + self.image_dir):
            for filename in filenames:
                self.filenames.append(dir + os.sep + filename)
        Frame.__init__(self, master)
        self.top_button_bar = Label(self)
        self.bind("<Configure>", self.resize)

        self.previous_butt = Button(self.top_button_bar, text="Previous Image", command=self.previous, anchor=W)
        self.previous_butt.pack(side=LEFT)   
        self.previous_and_save_butt = Button(self.top_button_bar, text="Save and Previous Image", command=self.previous_and_save, anchor=W)
        self.previous_and_save_butt.pack(side=LEFT)   
        self.next_butt = Button(self.top_button_bar, text="Next Image", command=self.next, anchor=W)
        self.next_butt.pack(side=LEFT)   
        self.next_and_save_butt = Button(self.top_button_bar, text="Save and Next Image", command=self.next_and_save, anchor=W)
        self.next_and_save_butt.pack(side=LEFT)
        self.save_datasets = Button(self.top_button_bar, text="Save Datasets", command=self.save_datasets, anchor=W)
        self.save_datasets.pack(side=LEFT)
        self.status_label = Label(anchor=W, justify=LEFT, textvariable=self.status_text)
        self.status_label.pack(side=RIGHT)

        self.top_button_bar.pack()     
        self.bottom_button_bar = Label(self)
        for entity in self.entities:
            self.image_presences[entity] = IntVar(value=0)
            self.entityCheckBoxes[entity] = Checkbutton(self.bottom_button_bar, text=entity, variable=self.image_presences[entity], anchor=W)
            self.entityCheckBoxes[entity].pack(side=LEFT)   
        self.bottom_button_bar.pack()     
        self.display = Canvas(self, bd=0, highlightthickness=0)
        self.display.pack()

        self.master = master
        from_db = H5ImageDatabase(os.getcwd()+os.sep+self.DB_FILE_NAME).load(self.entities)
        if (from_db):
            self.image_records = from_db[0]["training"][0]
        in_db = []
        for f in self.filenames:
            if (self.image_records and f in self.image_records.keys()):
                in_db.append(f)
        for f in in_db:
            self.filenames.remove(f)        
        self.filenames = in_db + self.filenames
        self.current_image_index = len(in_db) - 1
        self.init_window()

    def previous(self):
        self.previous_image(save=False)

    def previous_and_save(self):
        self.previous_image(save=True)

    def next(self):
        self.next_image(save=False)

    def next_and_save(self):
        self.next_image(save=True)

    # TODO get splitting into train, dev and dev working

    def save_datasets(self):
        # todo mv db to .db if it already exists
        datasets = {'training': [self.image_records, len(self.image_records)]}
        self.database = H5ImageDatabase(os.getcwd()+os.sep+self.DB_FILE_NAME)
        self.database.save(datasets)

    def _reset_image_presences(self):
        for entity in self.entities:
            self.image_presences[entity].set(0)

    def init_window(self):
        self.master.title("Manual Image Classifier")
        self.pack(fill=BOTH, expand=1)
        self.next_image(save=False)

    def _save_current_image_to_memory(self):
        self.image_records[self.current_filename] = ImageRecord()
        self.image_records[self.current_filename].cv2_image = self.cv2_image
        for entity in self.entities:
            self.image_records[self.current_filename].classifications[entity] = self.image_presences[entity].get()

    def previous_image(self, save=True):
        if (save):
            self._save_current_image_to_memory()
        self._reset_image_presences()
        self.current_image_index -= 1
        if (self.current_image_index == -1):
            self.current_image_index = len(self.filenames) - 1
        self._show_chosen_image()

    def next_image(self, save=True):
        if (save):
            self._save_current_image_to_memory()
        self._reset_image_presences()
        self.current_image_index += 1
        if (self.current_image_index == len(self.filenames)):
            self.current_image_index = 0
        self._show_chosen_image()
    
    def _show_chosen_image_from_memory(self, filename):
        record = self.image_records[filename]
        self._reset_image_presences()
        self.cv2_image = record.cv2_image
        self.converted_image = ImageTk.PhotoImage(Image.fromarray(self.cv2_image))
        for entity in self.entities:
            if (record.classifications[entity]):
                self.entityCheckBoxes[entity].select()
            else:
                self.entityCheckBoxes[entity].deselect()
        self.resize(None)

    def _show_chosen_image(self):
        self.current_filename = self.filenames[self.current_image_index]
        if (self.current_filename in self.image_records):
            self._show_chosen_image_from_memory(self.current_filename)
            return
        self._reset_image_presences()
        self.cv2_image = cv2.imread(self.current_filename)
        self.cv2_image = cv2.resize(self.cv2_image, (self.db_image_width, self.db_image_height), interpolation=cv2.INTER_CUBIC)
        self.cv2_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB)
        self.converted_image = ImageTk.PhotoImage(Image.fromarray(self.cv2_image))
        self.resize(None)

    def _get_summary_stats(self):
        stats = {"Unclassified": 0}
        for entity in self.entities:
            stats[entity] = 0
        for k in self.image_records.keys():
            classes = self.image_records[k].classifications
            total_classifications = 0
            for c in classes.keys():
                stats[c] += classes[c]
                total_classifications += classes[c]
            if (total_classifications == 0):
                stats["Unclassified"] +=1
        return stats

    def resize(self, event=None):
        if (event):
            self.height = event.height
            self.width = event.width
        self.display.config(width=self.width, height=self.height)
        self.image_on_canvas = self.display.create_image(0, 0, image=self.converted_image, anchor=NW, tags="IMG")
        short_fname = self.current_filename.split(os.sep)[-1]
        status_text = "File:  " + short_fname + '\n'
        stats = self._get_summary_stats()
        for s in stats.keys():
            status_text += s + ": " + str(stats[s]) + '\n'
        self.status_text.set(status_text)

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
db_image_height = config.get('database', 'imageHeight')
db_image_width = config.get('database', 'imageWidth')
image_dir = config.get('database', 'image_dir')
entities = config.get('general', 'enetitiesToLookFor').split(',')
app = Window(entities, image_dir, db_image_height, db_image_width, root)
root.mainloop()






