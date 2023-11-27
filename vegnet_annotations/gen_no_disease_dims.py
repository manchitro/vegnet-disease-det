import csv
import os
from PIL import Image

input_file = 'vegnet_annotations/vegnet-annots-all.csv'
output_file = 'vegnet_annotations/vegnet-annots-all-updated.csv'

with open(input_file, 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for row in rows:
        if 'No disease' in row[5]:
            file_name = row[5]
            file_path = os.path.join('vegnet_images', file_name)
            image = Image.open(file_path)
            width, height = image.size
            row.append(width)
            row.append(height)
        writer.writerow(row)