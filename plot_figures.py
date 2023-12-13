import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm


TNR = fm.FontProperties(fname='/home/eis/Workspace/Jaeyeon/Font/times.ttf')
    
plt.rc('font', size=15)        
plt.rc('axes', labelsize=20)   
plt.rc('xtick', labelsize=18)  
plt.rc('ytick', labelsize=18)  
plt.rc('legend', fontsize=13)  
plt.rc('figure', titlesize=20)
plt.rcParams["font.family"] = "Times New Roman"

datasets   = ["MIT-BIH Arr", "MIT-BIH ID","PAMAP2", "HAR-UCI", 'WiFi', 'Seizure', 'KETI']
algorithms = ["gzip", "CNN", "ResNet 18"]
accuracies = [
    [70.33, 12.96, 75.02, 51.35, 21.80, 68.60, 54.04],
    [34.40, 29.95, 88.80, 23.05, 44.20, 61.85, 65.74],
    [22.78, 90.96, 87.40, 25.00, 17.05, 70.08, 66.50],
]

datasets   = ["MIT-BIH Arr", "PAMAP2", 'WiFi']
algorithms = ["gzip", "CNN", "ResNet 18"]
accuracies = [
    [70.33,	92.51, 47.91],
    [34.46, 90.13, 44.20],
    [24.15, 89.10, 17.05],
]


cmap = plt.get_cmap('viridis_r')
num_dataset = len(datasets)
bar_width = 1/4

index = np.arange(num_dataset)
fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=400)
for i, algo in enumerate(algorithms):
    ax.bar(index+i*bar_width, accuracies[i], label=algo, width=bar_width, color=cmap((i)/len(algorithms)))

plt.yticks(np.arange(100, step=20), fontproperties=TNR)
plt.xticks(np.arange(bar_width, num_dataset + bar_width, 1), datasets, fontproperties=TNR)
ax.set_xlabel("Dataset", fontproperties=TNR)
ax.set_ylabel("Balanced accuracy (%)", fontproperties=TNR)
ax.set_ylim([0, 100])
ax.legend()
fig.tight_layout()
fig.savefig("./plots/samsung/exp1/b.png")
plt.close(fig)