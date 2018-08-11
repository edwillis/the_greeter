# the_greeter

A Machine-Learning based system to recognizing the people resident in a given house.
There are a few broad steps in deploying this project:

* Setting up a Raspberry Pi (or the platform of your choosing) to capture images in
the location you want to later recognize people from.
* Using the manual_classifier to classify the captured images into the people you've
specified in config.ini
* Training a neural network (on AWS, if you want to follow precisely what I did) to
rtecognize those people
* Deploying the trained model on that Raspberry Pi in the location you captured the
training images in so that people can be recognized as they walk through that area

## Capturing training set images
I used [raspbian] (https://www.raspberrypi.org/downloads/raspbian/) deployed on a
Raspberry Pi.  For image capture, I used a cheap web camera, the [Logitech C270]
(https://www.logitech.com/en-ca/product/hd-webcam-c270).  To configure the Pi, all
that's needed is to install and configure "motion" and "samba" (I've included my 
/etc/motion/motion.conf file and my /etc/samba/smb.conf files in the raspbian_config
directory to get you started - but note that the samba configuration shares an
unprotected directory containing the captured images, so revise accordingly if
you're not comfortable with that.  I also added these lines:

    sudo service motion start
    sudo motion

to /etc/rc.local.  Once you've done all this, reboot the Pi and you should start
seeing images populated to /var/lib/motion.  The configuration of motion results in
about 5 out of 6 images being not of people in my deployment - so changes in ambient
light or background motion triggers image capture in addition to people walking by
the camera.  This is expected and your experience may vary depending on the
characteristics of the location you're deploying into.  

## Using the manual_classifier to classify the captured images
See README.md in the scripts directory.

## Training a neural network 
I did this on AWS and will provide guidance on doing the same here.  There is no
inherent reason you would have to do this on AWS however - you could do it locally
following a similar procedure. 
To deploy training instances on AWS, I did the following:

* Visit [AWS] (https://aws.amazon.com/) and create an AWS account.  You'll need to
provide a payment method unless you want to try to train on the free tier EC2
instances (which I would not recommend). 
* Sign into the AWS console using your new credentials.
* Select "US - Virginia" as your region.
* Enter the S3 service in the console and create an S3 bucket (leave it with private
settings as we'll use an IAM role to allow access from our training EC2 instances to
the bucket).
* Upload your h5.db file (created by the manual_classifier script) to the S3 bucket,
as well as your config.ini file and the vgg_simple.py file.
* Go into the IAM service in the console.
* Create a role - I called mine "EC2S3Access" - and give that role the
"AmazonS3FullAccess" policy.
* Go into theEC2 service in the console.
* Launch an instance.  The key things to ensure here are:
  * Choose the instance type - I was pleased with the price and performance of the
"p2.xlarge" instances, but you are free to choose whatever you like.  I would not try
the low memory instances (< 16G or so) or anything without a GPU.
  * Choose the latest version of the "Deep Learning AMI (Ubuntu)" AMI as the OS for
the instance,
  * In "Configure Instance Details" make sure to set the "Shutdown behavior" to
"terminate" and the "IAM Role" to the role you created earlier.
  * Use the default storage options
  * You will be prompted to create an SSH key for accessing your instance.  Make sure
to save the resulting PEM file somewhere safe.
  * Make sure to select that SSH key when launching the instance by clicking on
"Review and Launch".
  * Once the instance is up and running (you'll see it in the EC2 service page in the
console), copy its IP address and ssh to it like so:  ssh -i <PATH TO PEM FILE>
ubuntu@<IP ADDRESS OF EC2 INSTANCE>.
  * Once you're on the instance, you can copy all the stuff from your S3 bucket like
so:  
  * aws s3 cp s3://<BUCKET NAME>/vgg_simple.py vgg_simple.py
  * aws s3 cp s3://<BUCKET_NAME>/config.ini config.ini
  * aws s3 cp s3://<BUCKET_NAME>/h5.db h5.db
  * And then run your training script like so:
  * source activate tensorflow_p36
  * python3  vgg_simple.py
  *Note that, for my images and classes (classifying four unique people using images
captured in my foyer) I was able to train up a CNN model in a couple of hours that
had false positives about 1 in 6 times it made a classification.
  *Note also that the vgg_simple.py script will leave a file containing the entire
trained model called "vggsimple_v1.model" in the current working directory - if you
want to keep it, a reasonable place to do so is in your S3 bucket - the local
storage on the EC2 instance is not persisted once the instance terminates so you
will lose the model if you don't copy this file somewhere.
