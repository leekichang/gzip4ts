import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm

    
plt.rc('font', size=15)        
plt.rc('axes', labelsize=20)   
plt.rc('xtick', labelsize=18)  
plt.rc('ytick', labelsize=18)  
plt.rc('legend', fontsize=13)  
plt.rc('figure', titlesize=20)

# datasets   = ["MIT-BIH Arr","PAMAP2", 'WiFi']
# algorithms = ["gzip", "RF", "AB", "CNN", "ResNet"]
# accuracies = [
#     [70.33, 12.96, 75.02],
#     [34.40, 29.95, 88.80],
#     [34.40, 29.95, 88.80],
#     [22.78, 90.96, 87.40],
#     [22.78, 90.96, 87.40],
# ]

datasets   = ["MIT-BIH Arr", "PAMAP2", 'WiFi']
algorithms = ["gzip", "RF", "AB", "CNN", "ResNet 18"]
accuracies = [
    [70.33,	92.51, 47.91],
    [62.23, 84.97, 53.80],
    [19.57, 40.58, 22.24],
    [34.40, 90.15, 88.80],
    [22.78, 90.96, 87.40],
]


cmap = plt.get_cmap('viridis_r')
num_dataset = len(datasets)
bar_width = 1/4

index = np.arange(num_dataset)
fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=400)
for i, algo in enumerate(algorithms):
    ax.bar(index+i*bar_width, accuracies[i], label=algo, width=bar_width, color=cmap((i)/len(algorithms)))

plt.yticks(np.arange(100, step=20))
plt.xticks(np.arange(bar_width, num_dataset + bar_width, 1), datasets)
ax.set_xlabel("Dataset")
ax.set_ylabel("Balanced accuracy (%)")
ax.set_ylim([0, 100])
ax.legend()
fig.tight_layout()
fig.savefig("./plots/samsung/exp1/b.png")
plt.close(fig)