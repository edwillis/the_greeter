# manual_classifier.py

## Description:
This script is used to create data for training, dev, and test sets.  The user
is guided through all the images in the images directory (specified in config.ini)
and annotates each image with the entities that are present in it.  The images are
reformatted to ensure a common size and the images as well as their
classifications, are written to an h5py database for later use in training.

## Usage:
Copy or rename scripts\config.ini.example to scripts\config.ini and edit it to meet
your needs.  The "image_dir" setting is reletive to the scripts directory.  The
database image height and width settings should match the aspect ratio of the bulk
of the images to classify and otherewise be big enough to yield images that don't
lose too much detail (you can view the converted images in the UI of the script
itself).  The entitiesToLookFor is a comma-separated list of the names corresponding
to the entities you wish to train the model to classify.  image_display_zoom indicates
how much to scale images up when displaying them in the UI - so if you are storing
the images in the file in smaller dimensions and find that hard to view when performing
classification, adjusting this value upwards will result in the images being displayed
larger which should make it easier.

If the database image height or width specified in the config.ini file do not match
the actual sizes of the images loaded from the database on script startup, then the
script will resize them to match the dimensions provided in config.ini.

When performing manual classification, the UI shows check boxes for each entity
specified in the config.ini file and so you can simply click on the names of the
people present in the image and then click "save and next" or "save and previous" to
record your selections - there are also keypad mappings employed so you can click the
number (counting the entity check boxes from left to right) and the the enter key to
accomplish the same.  Clicking on "save datasets" persists the sets to disk.  

You can review your classifications by entity by selecting the entity you wish from
the "View Only" entity selections.  Note you must unselect all "View Only" selections
to return to classifying new unclassified images.

To start the script, cd to the top level directory ("the_greeter") and run
"scripts/manual_classifier.py"
