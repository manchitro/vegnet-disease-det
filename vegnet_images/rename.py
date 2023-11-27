import os
import glob
import csv

mode = 'train' # 'val'
img_dir = "vegnet_images/"+mode
csv_file_path = "vegnet_annotations/vegnet-annots-"+mode+"-updated.csv"
csv_file_path_new = "vegnet_annotations/vegnet-annots-"+mode+"-updated-counted.csv"
# img_dir = "vegnet_images/val"
image_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.jpeg"))
image_files.sort()

# Read the CSV file
with open(csv_file_path, "r") as csvfile:
    csvreader = csv.reader(csvfile)
    rows = list(csvreader)

for i, image_file in enumerate(image_files):
    # print(i+1, image_file)
    new_image_file = os.path.join(img_dir, f"{i+1}_{os.path.basename(image_file)}")
    image_filename = os.path.basename(image_file)
    new_image_filename = os.path.basename(new_image_file)
    print(image_filename, new_image_filename)

    os.rename(image_file, new_image_file)

#     # Replace occurrences of image_file with new_image_file in each row of the CSV file
#     for row in rows:
#         if row[5] == image_filename:
#             row[5] = new_image_filename

# # Write the modified CSV file back
# with open(csv_file_path_new, "w") as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerows(rows)