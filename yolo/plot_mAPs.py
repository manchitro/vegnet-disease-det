# %%
import matplotlib.pyplot as plt
import pandas as pd
import os

def list_subdirectories(directories):
    subdirs = []
    for directory in directories:
        if os.path.isdir(directory):
            subdirs.extend([os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    return subdirs

directories = ['../vegnet_yolo_out']
# directories = ['../yolo_out', '../vegnet_yolo_out']
all_subdirs = list_subdirectories(directories)
all_subdirs.sort()
# all_subdirs = all_subdirs[0:7] + all_subdirs[9:]
# print(all_subdirs)
# List of csv files
# experiment_names = ['vegnet_yolov8s_original_dataset', 
#                     'vegnet_yolov8s_aug_200_all_cropped', 
#                     'vegnet_yolov8s_aug_200_210_all_cropped',
#                     'vegnet_yolov8s_200_210_untrained',
#                     'vegnet_yolov8s_200_210_untrained_cont',
#                     'vegnet_yolov8m_original2',
#                     'vegnet_yolov8m_200',
#                     'vegnet_yolov8m_200_210',
#                     'vegnet_yolov8m_200_210_220'
#                     ]
csv_files = [os.path.join(exp, 'results.csv') for exp in all_subdirs]

# Plotting the mAP progression for each csv file
fig, ax = plt.subplots(figsize=(10, 6))
i = 0
for file, label in zip(csv_files, all_subdirs):
    label = os.path.basename(label)
    df = pd.read_csv(file)
    mAP_values = df[df.columns[6]].tolist()
    max_mAP = max(mAP_values)
    idx_max = mAP_values.index(max_mAP)
    plt.plot(mAP_values, label=f"{i}_{label}")
    plt.ylim(0.0, 1.0)
    plt.annotate(
        # f"{label}\nMax mAP: {max_mAP:.2f}\nEpoch: {idx_max}",
        f"{i}_{max_mAP:.2f}",
        xy=(idx_max, max_mAP),
        xycoords="data",
        xytext=(0, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white"),
        fontsize=12,
    )
    i += 1
    

plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('mAP Progression Comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


all_subdirs = list_subdirectories(directories)
all_subdirs.sort()
for i, subdir in enumerate(all_subdirs):
    print(i, subdir)