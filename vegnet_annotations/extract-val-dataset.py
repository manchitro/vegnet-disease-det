import os
import csv

def check_csv_filenames(csv_file, directory, val_csv, train_csv):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        exist_count = 0
        not_exist_count = 0

        for row in reader:
            filename = row[5]
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(val_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                exist_count += 1
                print(f"File {filename} exists")
            else:
                with open(train_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                not_exist_count += 1
                print(f"File {filename} does not exist")

        print(f"Number of files that exist: {exist_count}")
        print(f"Number of files that do not exist: {not_exist_count}")

check_csv_filenames('/home/s/repo/vegnet-disease-det/vegnet_annotations/vegnet-annots-all.csv', '/home/s/Downloads/Datasets/Cauliflower Datasets/vegnet-train-val/Original Dataset/val/', '/home/s/repo/vegnet-disease-det/vegnet_annotations/vegnet-annots-val.csv', '/home/s/repo/vegnet-disease-det/vegnet_annotations/vegnet-annots-train.csv')