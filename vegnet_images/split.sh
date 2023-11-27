#!/bin/bash

# Path to the csv file
csv_file="./vegnet_annotations/vegnet-annots-val-updated.csv"

# Path to the vegnet_dir
vegnet_dir="./vegnet_images"

# Create the train subfolder
mkdir -p "${vegnet_dir}/val"

# Read the csv file line by line
while IFS=, read -r _ _ _ _ _ filename _ _
do
    # Check if the file exists in the vegnet_dir
	echo "${vegnet_dir}/${filename}"
    if [ -f "${vegnet_dir}/${filename}" ]; then
        # Move the file to the train subfolder
        mv "${vegnet_dir}/${filename}" "${vegnet_dir}/val/${filename}"
    fi
done < "$csv_file"
