import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8,5))

color = ['#993333', 'g', 'y', '#333399']

size = [11.7, 21.8, 25.6, 11.7]
accuracy = [58.9, 61.5, 63.8, 64.0] 
label = ['Basic(resnet18)','Basic(resnet34)','Basic(resnet50)', 'Ours(resnet18)']

for i in range(3):
    plt.scatter(size[i], accuracy[i], marker='x', color=color[i], label=label[i], s=100, linewidth=3)

plt.scatter(size[3], accuracy[3], marker='x', color=color[3], label=label[3], s=100, linewidth=3)

plt.plot(size[:3], accuracy[:3], linestyle='--', color='grey')
plt.plot([1.6, 11.7], [56.3, 58.9], linestyle='--', color='grey')
plt.plot([25.6, 29.4], [63.8, 66.1], linestyle='--', color='grey')

plt.xlabel('Parameter Size', fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)

plt.xlim(8, 30)
plt.ylim(55, 65)

plt.legend(bbox_to_anchor=(1, 0.8), prop={'size': 15})
plt.tight_layout()
    
#plt.show()    
plt.savefig('efficiency.pdf',format='pdf')