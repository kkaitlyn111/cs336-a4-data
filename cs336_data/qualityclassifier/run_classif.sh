#!/bin/bash

exec > /home/user/cs336-a4-data/classification_run.log 2>&1

uv run '/home/user/cs336-a4-data/cs336_data/qualityclassifier/gen_positive_paloma.py'
uv run '/home/user/cs336-a4-data/cs336_data/qualityclassifier/filter_positives.py'
uv run '/home/user/cs336-a4-data/cs336_data/qualityclassifier/train_classif.py'

