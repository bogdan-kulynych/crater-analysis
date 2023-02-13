#!/usr/bin/env bash

unzip 'job_5-2023_01_30_20_47_52-imagenet 1.0.zip' -d train
unzip 'job_6-2023_01_26_14_45_07-imagenet 1.0.zip' -d train
unzip 'job_7-2023_01_25_17_54_55-imagenet 1.0.zip' -d train
unzip 'job_8-2023_01_27_13_24_37-imagenet 1.0.zip' -d test

mv 'train/bombed' train/1
mv 'train/not bombed' train/0
mv 'test/bombed' test/1
mv 'test/not bombed' test/0
