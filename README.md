# Facial Recognition

## Installation

Make sure you have python3 already installed on your system, then run below command to install required libraries


```bash
python3 -m pip install -r requirements.txt
```

## Usage

***matcher.py*** This script compares the faces in two images, and responds by drawing a box ad showing whether or not the two images contain the same person.

>NOTE: Make sure the two images only contain one face per image

Below are the required parameters to be passed to the script

```bash
python3 matcher.py -t images/arthur.jpg -s images/kay.jpg
```

Where the `-t` is the image to test, and `-s` is the image to use to train model on facial landmarks


***video.py*** This script esssentially requires at least two images of two different people. THe script runs a rela time frame by frame inference to see if any of the faces in the video feeds are the same as the ones use in the encodings to be searched.

To execute this, run below command.

```bash
python3 video.py
```


Happy Coding...
