import numpy as np
import csv
import matplotlib.pyplot as plt

num_epochs = 30

fig = plt.figure(figsize=(13,6))
fig.subplots_adjust(left=0.07, bottom=0.13, right=0.95, top=0.83)

color = ['y', '#993333','g', '#333399']

for j, experiment in enumerate(['basic_resnet18', 'basic_resnet34', 'basic_resnet50', 'generative_vqa']):

    for phase in ['train', 'valid']:
        
        epoch = []
        loss = []
        acc = []
        
        for i in range(num_epochs):
            
            with open('./logs/{}/{}-log-epoch-{:02d}.txt'.format(experiment, phase, i+1), 'r') as f:
                df = csv.reader(f, delimiter='\t')
                data = list(df)

            epoch.append(float(data[0][0]))
            loss.append(float(data[0][1]))
            acc.append(float(data[0][3]))

        plt.subplot(1, 2, 1)
        if phase == 'train':
            plt.plot(epoch, loss, label = experiment + ' ' + phase, color = color[j], linewidth = 3.0, linestyle='--')
        else:
            plt.plot(epoch, loss, label = experiment + ' ' + phase, color = color[j], linewidth = 3.0, linestyle='-')
                
        plt.xlabel('Epoch', fontsize = 20)
        plt.ylabel('Loss', fontsize = 20)
            
        plt.subplot(1, 2, 2)
        #plt.tight_layout()

        if phase == 'train':
            plt.plot(epoch, acc, label = experiment + ' ' + phase, color = color[j], linewidth = 3.0, linestyle='--')
        else:
            plt.plot(epoch, acc, label = experiment + ' ' + phase, color = color[j], linewidth = 3.0, linestyle='-')
        
        plt.xlabel('Epoch', fontsize = 20)
        plt.ylabel('Accuracy', fontsize = 20)
        

        print(experiment, phase, max(acc))

plt.legend(loc='upper center', bbox_to_anchor=(-0.1, 1.25), prop={'size': 15}, ncol=4)

#plt.show()    
plt.savefig('train.pdf',format='pdf')