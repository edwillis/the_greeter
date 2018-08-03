manual_classifier.py

Description:
This script is used to create data for training, dev, and test sets.  The user
is guided through all the images in the images directory and annotates each image
with the entities that are present in it.  The images are reformatted to ensure a
common size and the images as well as their classifications, are written to an
h5py database for later use in training.

Usage:
Copy or rename scripts\config.ini.example to scripts\config.ini and edit it to meet
your needs.  The "image_dir" setting is reletive to the scirpts directory.  The
database image height and width settings should match the aspect ratio of the bulk
of the images to classify and otherewise be big enough to yield images that don't
lose too much detail (you can view the converted images in the UI of the script
itself).  The entitiesToLookFor is a comma-separated list of the names corresponding
to the entities you wish to train the model to classify.  

If the database image height or width specified in the config.ini file do not match
the actual sizes of the images loaded from the database on script startup, then the
script will resize them to match the dimensions provided in config.ini.

To start the script, cd to the top level directory ("the_greeter") and run
"scripts/manual_classifier.py"
