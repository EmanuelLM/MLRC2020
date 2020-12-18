import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt


! git clone https://github.com/ElementAI/synbols-benchmarks.git

# enter into classification (supervised learning) experiment
os.chdir('/content/drive/MyDrive/synbols-benchmarks/classification')
!pip install -r requirements.txt


files = [
         '/content/drive/MyDrive/resNet12+/c4797e3a55ec84551bbbc7d6cdfdd051', #default100k - aug=True
         '/content/drive/MyDrive/resNet12+/16b22835f86a1fab2a810a7d00186519', #camouflage - aug=True
         '/content/drive/MyDrive/resNet12+_2/a48feb801cd91f7de691cf15611effd9', #korean - aug=True
         '/content/drive/MyDrive/resNet12_less_var_aug=True/e2b24b404389c790d2a9ac7e5bd348ce', #less_variation - aug=True
         '/content/drive/MyDrive/resNet12+/41810817d5c7cdb18ba665e6b0b58752', #default100k - aug=False
         '/content/drive/MyDrive/resNet12+/529536036661a1920471ea13728260bd', #camouflage - aug=False
         '/content/drive/MyDrive/resNet12+_korean_aug=false/244259469440de8cc1f98e25758cca1a',   #korean - aug=False
         '/content/drive/MyDrive/resNet12_less_var_aug=False/7fe4fd04d51d442bf5189585dfdb83ed' #less_variation - aug= False
         ]

task_genre =[
             'default100k_aug',
             'camouflage_aug',
             'korean_aug',
             'lessvar_aug',
             'default100k',
             'camouflage',
             'korean',
             'lessvar'
             ]

for (i, task) in zip(files, task_genre):
  add_path = f'{i}/score_list.pkl'
  globals()['train_loss_' + 'ResNet_' + task] = []
  globals()['valid_loss_' + 'ResNet_' + task] = []
  with open(add_path, 'rb') as temp_file:
    temp_data = pkl.load(temp_file)
    for j in temp_data:
        globals()['train_loss_' + 'ResNet_' + str(task)].append(j['train_loss'])
        globals()['valid_loss_' + 'ResNet_' + str(task)].append(j['val_loss'])


# plot for train_loss and valid_loss series 1
plt.figure(figsize=(10,7))

plt.plot(train_loss_ResNet_default100k_aug, label='train-default100k-aug', linestyle='-.', color='orange')
plt.plot(valid_loss_ResNet_default100k_aug, label='valid-default100k-aug', linestyle='solid', color='orange', marker='x', markersize=5)

plt.plot(train_loss_ResNet_default100k, label='train-default100k', linestyle='-.', color='xkcd:lime green')
plt.plot(valid_loss_ResNet_default100k, label='valid-default100k', linestyle='solid', color='xkcd:lime green', marker='x', markersize=5)



plt.plot(train_loss_ResNet_camouflage_aug, label='train-camouflage-aug', linestyle='-.', color='xkcd:cyan')
plt.plot(valid_loss_ResNet_camouflage_aug, label='valid-camouflage-aug', linestyle='solid', color='xkcd:cyan', marker='x', markersize=5)

plt.plot(train_loss_ResNet_camouflage, label='train-camouflage', linestyle='-.', color='xkcd:purple')
plt.plot(valid_loss_ResNet_camouflage, label='valid-camouflage', linestyle='solid', color='xkcd:purple', marker='x', markersize=5)


plt.title('Cross-entropy loss for ResNet-12 in default and camouflage tasks')
plt.ylabel('train and valid loss')

plt.xlabel('Number of epoch')
plt.legend(loc="upper right")
plt.show()


# plot for train_loss and valid_loss series 2
plt.figure(figsize=(10,7))

plt.plot(train_loss_ResNet_korean_aug, label='train-korean-aug', linestyle='-.', color='xkcd:red')
plt.plot(valid_loss_ResNet_korean_aug, label='valid-korean-aug', linestyle='solid', color='xkcd:red', marker='x', markersize=5)

plt.plot(train_loss_ResNet_korean, label='train-korean', linestyle='-.', color='xkcd:grass green')
plt.plot(valid_loss_ResNet_korean, label='valid-korean', linestyle='solid', color='xkcd:grass green', marker='x', markersize=5)



plt.plot(train_loss_ResNet_lessvar_aug, label='train-lessvar-aug', linestyle='-.', color='xkcd:blue')
plt.plot(valid_loss_ResNet_lessvar_aug, label='valid-lessvar-aug', linestyle='solid', color='xkcd:blue', marker='x', markersize=5)

plt.plot(train_loss_ResNet_lessvar, label='train-lessvar', linestyle='-.', color='xkcd:fuchsia')
plt.plot(valid_loss_ResNet_lessvar, label='valid-lessvar', linestyle='solid', color='xkcd:fuchsia', marker='x', markersize=5)


plt.title('Cross-entropy loss for ResNet-12 in korean and less variation tasks')
plt.ylabel('train and valid loss')

plt.xlabel('Number of epoch')
plt.legend(loc="upper right")
plt.show()


# print test accuracy
print('<<test accuracy for ResNet-12>>')
for i, task in zip(files, task_genre):
  add_path = f'{i}/score_list_test.pkl'
  with open(add_path, 'rb') as temp_file:
    temp_data = pkl.load(temp_file)
    print('test accuracy for ' +  task + ': ' + str(temp_data['test_accuracy']))

