#!/usr/bin/env bash

cd data/
echo "Downloading the Oxford hand dataset ..."
wget http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz
echo "Download completed."

tar -xf hand_dataset.tar.gz
ln -s hand_dataset/evaluation_code/VOC2007/VOCdevkit VOCdevkit
python scripts/create_trainval.py

