! git clone https://github.com/ElementAI/synbols-benchmarks.git
import os
import sys
import pickle as pkl
from google.colab import drive


drive.mount('/content/drive')
%cd /content/drive/MyDrive/
os.chdir('/content/drive/MyDrive/synbols-benchmarks/classification')
!pip install -r requirements.txt


! python3 trainval.py --savedir_base '/content/drive/MyDrive/WRN' --exp_group_list baselines


files = [         
         '/content/drive/MyDrive/WRN/Camouflage', 
         '/content/drive/MyDrive/WRN2/Camouflage(+)', 
         '/content/drive/MyDrive/WRN/Korean', 
         '/content/drive/MyDrive/WRN2/LessVar(+)', 
         '/content/drive/MyDrive/WRN2/Korean(+)',
         '/content/drive/MyDrive/WRN/LessVar', 
         '/content/drive/MyDrive/WRN/default100k'
         '/content/drive/MyDrive/WRN2/default100k(+)'
         ]

task_genre =['camouflage',
             'Camouflage(+)',
             'Korean(+)',
             'korean',
             'lessVar',
             'LessVar(+)',
             'default100k',
             'default100k(+)'
             ] 

for (i, task) in zip(files, task_genre):
  add_path = f'{i}/score_list.pkl'
  globals()['train_loss_' + 'WRN_' + task] = []
  globals()['val_loss_' + 'WRN_' + task] = []
  with open(add_path, 'rb') as temp_file:
    temp_data = pkl.load(temp_file)
    for j in temp_data:
        globals()['train_loss_' + 'WRN_' + str(task)].append(j['train_loss'])
        globals()['val_loss_' + 'WRN_' + str(task)].append(j['val_loss'])




plt.figure(figsize=(9,5))

plt.plot(train_loss_WRN_default100k, label='train-default100k', linestyle='-.', color='orange')
plt.plot(val_loss_WRN_default100k, label='valid-default100k', linestyle='solid', color='orange', marker='x', markersize=5)


plt.plot(train_loss_WRN_camouflage, label='train-camouflage', linestyle='-.', color='xkcd:cyan')
plt.plot(val_loss_WRN_camouflage, label='valid-camouflage', linestyle='solid', color='xkcd:cyan', marker='x', markersize=5)


plt.plot(train_loss_WRN_korean, label='train-korean', linestyle='-.', color='xkcd:red')
plt.plot(val_loss_WRN_korean, label='valid-korean', linestyle='solid', color='xkcd:red', marker='x', markersize=5)


plt.ylabel('Train and Valid loss')

plt.xlabel('Number of epoch')
plt.legend(loc="upper right")
plt.show()
