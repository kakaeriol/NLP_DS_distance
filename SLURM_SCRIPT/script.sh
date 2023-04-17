#!/bin/bash
if [ -f "/home/n/nguyenpk/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/home/n/nguyenpk/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="/home/n/nguyenpk/miniconda3/bin:$PATH"
fi
conda activate otdd_tf

python_path=/home/n/nguyenpk/CS6220/project/NLP_DS_distance/DD_conv_distance_topic.py

python -u ${python_path} --id $1

