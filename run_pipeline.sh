#!/bin/sh

python source_code/preprocess.py
python source_code/train_classifier.py
python source_code/train_segmentation.py
python source_code/evaluate.py