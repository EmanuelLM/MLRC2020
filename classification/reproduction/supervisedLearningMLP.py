! git clone https://github.com/ElementAI/synbols-benchmarks.git
import os
import sys
import pickle as pkl
from google.colab import drive


drive.mount('/content/drive')
%cd /content/drive/MyDrive/
os.chdir('/content/drive/MyDrive/synbols-benchmarks/classification')
!pip install -r requirements.txt


! python3 trainval.py --savedir_base '/content/drive/MyDrive/MLP' --exp_group_list baselines


files = [
         '/content/drive/MyDrive/MLP/Camouflage(+)', 
         '/content/drive/MyDrive/MLP/Korean(+)', 
         '/content/drive/MyDrive/MLP/LessVar(+)', 
         '/content/drive/MyDrive/MLP/default100k(+)' 
         '/content/drive/MyDrive/MLP/Camouflage', 
         '/content/drive/MyDrive/MLP/Korean', 
         '/content/drive/MyDrive/MLP/LessVar', 
         '/content/drive/MyDrive/MLP/default100k'
         ]

task_genre =['Camouflage',
             'Korean',
             'LessVar',
             'default100k'
             ] 

for (i, task) in zip(files, task_genre):
  add_path = f'{i}/score_list_test.pkl'
  with open(add_path, 'rb') as temp_file:
    temp_data = pkl.load(temp_file)
    test_acc = temp_data['test_accuracy']
    print(f'the test_accruacy for {task}¨is {test_acc}')

files = [         
         '/content/drive/MyDrive/MLP2/default100k(3)', #default100k
         '/content/drive/MyDrive/MLP2/Camouflage(3)', #camouflage
         '/content/drive/MyDrive/MLP2/korean(3)', #korean
         '/content/drive/MyDrive/MLP2/LessVar100k(3)' #less_variation
         ]

task_genre =['default100k',
             'camouflage',
             'korean',
             'lessvar'
	     'default100k+',
             'camouflage+',
             'korean+',
             'lessvar+'
             ] 

for (i, task) in zip(files, task_genre):
  add_path = f'{i}/score_list.pkl'
  globals()['train_loss_' + 'MLP_' + task] = []
  globals()['valid_loss_' + 'MLP_' + task] = []
  with open(add_path, 'rb') as temp_file:
    temp_data = pkl.load(temp_file)
    for j in temp_data:
        globals()['train_loss_' + 'MLP_' + str(task)].append(j['train_loss'])
        globals()['valid_loss_' + 'MLP_' + str(task)].append(j['val_loss'])


plt.figure(figsize=(10,7))

plt.plot(train_loss_MLP_default100k, label='train-default100k', linestyle='-.', color='orange')
plt.plot(valid_loss_MLP_default100k, label='valid-default100k', linestyle='solid', color='orange', marker='x', markersize=5)


plt.plot(train_loss_MLP_camouflage, label='train-camouflage', linestyle='-.', color='xkcd:cyan')
plt.plot(valid_loss_MLP_camouflage, label='valid-camouflage', linestyle='solid', color='xkcd:cyan', marker='x', markersize=5)


plt.plot(train_loss_MLP_korean, label='train-korean', linestyle='-.', color='xkcd:red')
plt.plot(valid_loss_MLP_korean, label='valid-korean', linestyle='solid', color='xkcd:red', marker='x', markersize=5)

plt.plot(train_loss_MLP_lessvar, label='train-lessvar', linestyle='-.', color='xkcd:blue')
plt.plot(valid_loss_MLP_lessvar, label='valid-lessvar', linestyle='solid', color='xkcd:blue', marker='x', markersize=5)

plt.title('Cross-entropy loss for MLP in different tasks')
plt.ylabel('train and valid loss')

plt.xlabel('Number of epoch')
plt.legend(loc="upper right")
plt.show()