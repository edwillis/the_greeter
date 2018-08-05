import os
from tkinter import *
from PIL import Image,ImageTk
from pprint import pprint
import numpy as np
import h5py
import cv2
import configparser
import random
import math

class H5ImageDatabase():

    def __init__(self, filename):
        self.filename = filename

    def save(self, datasets, image_height, image_width):
        if (not datasets):
            return
        self.hdf5_file = h5py.File(self.filename, mode='w')
        for dataset in datasets:
            if (not dataset):
                continue
            grp = self.hdf5_file.create_group(dataset)
            image_recs = datasets[dataset][1]
            fname_recs = []
            fname_recs = [ ir[0] for ir in image_recs ]
            fname_recs = [r.encode("ascii", "ignore") for r in fname_recs]
            # persist the image filenames
            grp.create_dataset("filenames", data=fname_recs, dtype=h5py.special_dtype(vlen=str))
            # persist the images themselves (ML inputs)
            recs = [ ir[1] for ir in image_recs ]
            img_recs = [ i.cv2_image for i in recs]
            img_rec_shape = (len(img_recs), img_recs[0].shape[0], img_recs[0].shape[1], img_recs[0].shape[2])
            img_dataset = grp.create_dataset("inputs", img_rec_shape, np.float16, chunks=(10, img_recs[0].shape[0], img_recs[0].shape[1], 3))
            for i in range(len(img_recs)):
                img_dataset[i, ...] = img_recs[i][None]/255.0
            # persist the classifications (ML outputs)
            class_recs = [ i.classifications for i in recs]
            class_rec_shape = (len(class_recs), len(class_recs[0]))
            cls_dataset = grp.create_dataset("outputs", class_rec_shape, np.uint8)
            for i in range(len(class_recs)):
                classes = list(class_recs[i].values())
                cls_dataset[i] = classes
        self.hdf5_file.close()

    def load(self, entities, expected_height, expected_width):
        try:
            combined_datasets = {}
            self.hdf5_file = h5py.File(self.filename, mode='r')
            for group in self.hdf5_file:
                combined_datasets[group] = dict()
                conversion = np.rint(self.hdf5_file[group]["inputs"].value * 255).astype(np.uint8)
                zips = zip(self.hdf5_file[group]["filenames"],
                           conversion,
                           self.hdf5_file[group]["outputs"].value)
                for fname, input, output in zips:
                    combined_datasets[group][fname] = ImageRecord()
                    combined_datasets[group][fname].cv2_image = input
                    if (combined_datasets[group][fname].cv2_image.shape[0] != expected_height or
                        combined_datasets[group][fname].cv2_image.shape[1] != expected_width):
                        combined_datasets[group][fname].cv2_image = cv2.resize(combined_datasets[group][fname].cv2_image, (expected_width, expected_height), interpolation=cv2.INTER_CUBIC)
                    cls_pairs = zip(entities, output)
                    for c in cls_pairs:
                        combined_datasets[group][fname].classifications[c[0]] = c[1]
            self.hdf5_file.close()
            return combined_datasets
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

    def __init__(self, entities, image_dir, db_image_height, db_image_width, percentage_training, percentage_dev, percentage_test, master=None):
        self.current_image_index = -1
        self.image = None
        self.converted_image = None
        self.image_on_canvas = None
        self.height = self.INITIAL_HEIGHT
        self.width = self.INITIAL_WIDTH
        self.image_dir="images"
        self.db_image_height = db_image_height
        self.db_image_width = db_image_width
        self.percentage_training = percentage_training
        self.percentage_dev = percentage_dev
        self.percentage_test = percentage_test
        self.entities = entities
        self.entityCheckBoxes = {}
        self.image_presences = {}
        self.database = None
        self.status_text=StringVar()
        self.image_records = {}
        self.filenames=[]
        self.sets = None
        self.entity_map = None
        self.current_filename=None
        self.wrapped = False
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
        from_db = H5ImageDatabase(os.getcwd()+os.sep+self.DB_FILE_NAME).load(self.entities, self.db_image_height, self.db_image_width)
        if (from_db):
            self.image_records = dict()
            for group in from_db:
                self.image_records.update(from_db[group])
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

    def save_datasets(self):
        # todo mv db to .db if it already exists

        # make a dictionary that maps entities to the list of indices into 
        # the image records where images were classified as that entity
        self.entity_map = dict()
        self.entity_map["Unclassified"] = list()
        for entity in self.entities:
            self.entity_map[entity] = list()
        for ir in self.image_records.keys():
            classification_count = 0
            for entity in self.entities:
                if (self.image_records[ir].classifications[entity] > 0):
                    self.entity_map[entity].append(ir)
                    # first one wins - if a record has multiple classifications
                    # we treat it as belonging to the first class we see
                    break
                classification_count+=1
            if (classification_count == len(self.entities)):
                self.entity_map["Unclassified"].append(ir)

        # shuffle each list of indices in the dictionary
        for entity in self.entity_map:
            if (self.entity_map[entity]):
                random.shuffle(self.entity_map[entity])

        # make a dictionary that maps the test set names to a slice of each
        # per-entity index list that's as long as the precentages require
        self.sets = dict()
        self.sets['dev'] = [ self.percentage_dev, [] ]
        self.sets['test'] = [ self.percentage_test, [] ]
        self.sets['training'] = [ self.percentage_training, [] ]

        for entity in self.entity_map:
            recs_taken = 0
            for s in self.sets:
                to_take = min(math.ceil((self.sets[s][0]/100.0) * len(self.entity_map[entity])), len(self.entity_map[entity]))
                # the ceiling calls round up for dev and test sets, so we would
                # overcount on the training set unless we take the min of the sum
                # and the actual end of the list
                upper_end = min(recs_taken+to_take, len(self.entity_map[entity]))
                # self.sets[s][1] += self.entity_map[entity][recs_taken:upper_end]
                for ir in self.entity_map[entity][recs_taken:upper_end]:
                    self.sets[s][1].append([ir, self.image_records[ir]])
                recs_taken += to_take

        # concatenate all the per-training set lists and shuffle them
        for s in self.sets:
            if (self.sets[s]):
                random.shuffle(self.sets[s][1])
        self.resize()

        self.database = H5ImageDatabase(os.getcwd()+os.sep+self.DB_FILE_NAME)
        self.database.save(self.sets, self.db_image_height, self.db_image_width)

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
            self.wrapped = True
        self._show_chosen_image()

    def next_image(self, save=True):
        if (save):
            self._save_current_image_to_memory()
        self._reset_image_presences()
        self.current_image_index += 1
        if (self.current_image_index == len(self.filenames)):
            self.current_image_index = 0
            self.wrapped = True
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
            total_classifications_curr_record = 0
            for c in classes.keys():
                stats[c] += classes[c]
                total_classifications_curr_record += classes[c]
            if (total_classifications_curr_record == 0):
                stats["Unclassified"] +=1
        stats["Total"] =len(self.image_records)
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

        if (self.entity_map and self.sets):
            status_text += "\nTraining/dev/test sets composition:\n"
            for entity in self.entity_map:
                recs_taken = 0
                for s in self.sets:
                    to_take = min(math.ceil((self.sets[s][0]/100.0) * len(self.entity_map[entity])), len(self.entity_map[entity]))
                    if (recs_taken + to_take > len(self.entity_map[entity])):
                        to_take = len(self.entity_map[entity]) - recs_taken
                    status_text += entity + ", " + s + ": percentage " + str(self.sets[s][0]) + " total records " + str(to_take) + '\n'
                    recs_taken += to_take
            for s in self.sets:
                status_text += "Total in " + s + ":  " + str(len(self.sets[s][1])) + '\n'
        if (self.wrapped):
            status_text += "\nWRAPPED"
            self.wrapped = False

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
config.read('config.ini')
db_image_height = config.getint('database', 'imageHeight')
db_image_width = config.getint('database', 'imageWidth')
image_dir = config.get('database', 'image_dir')
entities = config.get('general', 'entitiesToLookFor').split(',')
percentage_training = config.getint('general', "percentage_training")
percentage_dev = config.getint('general', "percentage_dev")
percentage_test = config.getint('general', "percentage_test")
if (percentage_training + percentage_dev + percentage_test != 100):
    print("training, dev and test set percentages do notsum to 100 - check config.ini")
    exit(1)
app = Window(entities, image_dir, db_image_height, db_image_width, percentage_training, percentage_dev, percentage_test, root)
root.mainloop()






