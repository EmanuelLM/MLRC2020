import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt


! git clone https://github.com/ElementAI/synbols-benchmarks.git

# enter into classification (supervised learning) experiment
os.chdir('/content/drive/MyDrive/synbols-benchmarks/classification')
!pip install -r requirements.txt

# reproduce the results with seed 3
! python3 trainval.py --savedir_base '/content/drive/MyDrive/CONV-4-FLAT' --exp_group_list baselines

files = [
         '/content/drive/MyDrive/CONV-4-flat/fe65a608f009d871c0d4696bb589f1da', #default100k
         '/content/drive/MyDrive/CONV-4-flat/5bdf76f271ce8d3925fb64f2dda7792c', #camouflage
         '/content/drive/MyDrive/CONV-4-flat/ce04601c8ce08d5d3811192734946d3c', #korean
         '/content/drive/MyDrive/CONV-4-flat/17e2cb4b42b8d6c0e80fcff5ccce1a4c' #less_variation
         ]


colors = [
          'orange',
          'xkcd:cyan',
          'xkcd:red',
          'xkcd:blue'
          ]

for (i, task) in zip(files, task_genre):
  add_path = f'{i}/score_list.pkl'
  globals()['train_loss_' + 'conv4FLAT_' + task] = []
  globals()['valid_loss_' + 'conv4FLAT_' + task] = []
  with open(add_path, 'rb') as temp_file:
    temp_data = pkl.load(temp_file)
    for j in temp_data:
        globals()['train_loss_' + 'conv4FLAT_' + str(task)].append(j['train_loss'])
        globals()['valid_loss_' + 'conv4FLAT_' + str(task)].append(j['val_loss'])


# plot for train_loss and valid_loss
plt.figure(figsize=(10,7))

for task, color in zip(task_genre, colors):
    plt.plot(globals()['train_loss_' + 'conv4FLAT_' + str(task)], label= 'train' + '-' + str(task),
             linestyle='-.', color= color)
    plt.plot(globals()['valid_loss_' + 'conv4FLAT_' + str(task)], label='train' + '-' + str(task),
             linestyle='solid', color=color, marker='x', markersize=5)

plt.title('Cross-entropy loss for conv-4-FLAT in different tasks')
plt.ylabel('train and valid loss')

plt.xlabel('Number of epoch')
plt.legend(loc="upper right")
plt.show()


# print test accuracy
print('<<test accuracy for CONV4-FLAT>>')
for i, task in zip(files, task_genre):
  add_path = f'{i}/score_list_test.pkl'
  with open(add_path, 'rb') as temp_file:
    temp_data = pkl.load(temp_file)
    print('test accuracy for ' +  task + ': ' + str(temp_data['test_accuracy']))