#!/bin/bash

# Check if the minimum segment size is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <min_segment_size>"
    exit 1
fi

# Define the range of threshold values
for i in 0.2 0.25 0.3 0.35 0.4 0.45 0.5
do
    # Run the Python script with the corresponding threshold and minimum segment size
    python graphseg_timer.py --input ~/Downloads/wiki_dev_100_np_seperators \
                             --output ~/Downloads/wiki_dev_100_np_seperators_output \
                             --jar graphseg.jar \
                             --threshold $i \
                             --min_segment $1
done