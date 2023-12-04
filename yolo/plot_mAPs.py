# %%
import matplotlib.pyplot as plt
import pandas as pd
import os

# List of csv files
experiment_names = ['vegnet_yolov8s_original_dataset', 
                    'vegnet_yolov8s_aug_200_all_cropped', 
                    'vegnet_yolov8s_aug_200_210_all_cropped',
                    'vegnet_yolov8s_200_210_untrained',
                    'vegnet_yolov8s_200_210_untrained_cont',
                    'vegnet_yolov8m_original2',
                    'vegnet_yolov8m_200',
                    'vegnet_yolov8m_200_210',
                    'vegnet_yolov8m_200_210_220'
                    ]
csv_files = [os.path.join('../yolo_out', experiment, 'results.csv') for experiment in experiment_names]
print('cwd:', os.getcwd())

# Plotting the mAP progression for each csv file
fig, ax = plt.subplots(figsize=(10, 6))
for file, label in zip(csv_files, experiment_names):
    df = pd.read_csv(file)
    mAP_values = df[df.columns[6]].tolist()
    max_mAP = max(mAP_values)
    idx_max = mAP_values.index(max_mAP)
    plt.plot(mAP_values, label=label)
    plt.annotate(
        # f"{label}\nMax mAP: {max_mAP:.2f}\nEpoch: {idx_max}",
        f"{max_mAP:.2f}",
        xy=(idx_max, max_mAP),
        xycoords="data",
        xytext=(0, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white"),
        fontsize=12,
    )
    

plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('mAP Progression Comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

