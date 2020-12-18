import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt


! git clone https://github.com/ElementAI/synbols-benchmarks.git

# enter into classification (supervised learning) experiment
os.chdir('/content/drive/MyDrive/synbols-benchmarks/classification')
!pip install -r requirements.txt

# reproduce the results with seed 3
! python3 trainval.py --savedir_base '/content/drive/MyDrive/CONV-4-GAP' --exp_group_list baselines


### plot for train loss and val loss

#---- conv-4-GAP ----#
#change the file path to your customized path
files = [
         '/content/drive/MyDrive/conv_4_GAP/a5728d39488d9b5572267b8825652a69', #default100k
         '/content/drive/MyDrive/conv_4_GAP/1494ba72bf70b3e324ed26090c4f71a8', #camouflage
         '/content/drive/MyDrive/conv_4_GAP/bcd2870382d348402a0ff290f04fa824', #korean
         '/content/drive/MyDrive/conv_4_GAP/d772582b13d58e435fd2f54f7bc0ec82' #less_variation
         ]

task_genre =['default100k',
             'camouflage',
             'korean',
             'lessvar'
             ]
colors = [
          'orange',
          'xkcd:cyan',
          'xkcd:red',
          'xkcd:blue'
          ]

for (i, task) in zip(files, task_genre):
  add_path = f'{i}/score_list.pkl'
  globals()['train_loss_' + 'conv4GAP_' + task] = []
  globals()['valid_loss_' + 'conv4GAP_' + task] = []
  with open(add_path, 'rb') as temp_file:
    temp_data = pkl.load(temp_file)
    for j in temp_data:
        globals()['train_loss_' + 'conv4GAP_' + str(task)].append(j['train_loss'])
        globals()['valid_loss_' + 'conv4GAP_' + str(task)].append(j['val_loss'])




# plot for train_loss and valid_loss
plt.figure(figsize=(10,7))

for task, color in zip(task_genre, colors):
    plt.plot(globals()['train_loss_' + 'conv4GAP_' + str(task)], label= 'train' + '-' + str(task),
             linestyle='-.', color= color)
    plt.plot(globals()['valid_loss_' + 'conv4GAP_' + str(task)], label='train' + '-' + str(task),
             linestyle='solid', color=color, marker='x', markersize=5)

plt.title('Cross-entropy loss for conv-4-GAP in different tasks')
plt.ylabel('train and valid loss')

plt.xlabel('Number of epoch')
plt.legend(loc="upper right")
plt.show()



# print test accuracy
print('<<test accuracy for CONV4-GAP>>')
for i, task in zip(files, task_genre):
  add_path = f'{i}/score_list_test.pkl'
  with open(add_path, 'rb') as temp_file:
    temp_data = pkl.load(temp_file)
    print('test accuracy for ' +  task + ': ' + str(temp_data['test_accuracy']))
