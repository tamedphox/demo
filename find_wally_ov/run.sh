#!/bin/bash 

if [ -z "$1" ]; then
	echo "Usage : $0 <image_path>"
	exit 1
fi

IMAGE_PATH="$1"

python3 ./find_wally_ov.py \
       --model ./wally.xml \
       --image "$IMAGE_PATH" \
       --labels ./label.txt \
       --score-thresh 0.9

