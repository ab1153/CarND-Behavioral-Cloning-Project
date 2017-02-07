#!/bin/bash
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock
sudo apt get install zip
aws s3 cp $1/data.zip ./
unzip ./data.zip