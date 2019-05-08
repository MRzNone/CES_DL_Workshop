import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import glob, os
# import scipy.misc
from PIL import Image
import numpy as np
import random
# import matplotlib.pyplot as plt

shrink_size = (70,70)
data = []

'''
  Step 1: Load Data
'''
# read data sets
print("loading data sets")
os.chdir("./shapes")
# os.chdir("./small")

label = 0
for path in glob.glob("*"):
    # Folders
    for file in glob.glob(path + "/**"):
      img = Image.open(file)
      img.thumbnail(shrink_size)
      img = img.convert("L")
      
      data.append({
        "label": label,
        "img": np.asarray(img, dtype=np.float32)
        })
    label = label + 1
    print(path + " done")
  

'''
  Step 2: Make Data iterable
'''

# separate train data and test data

random.shuffle(data)

num_test = 3000
test_data = data[0:num_test]
train_data  = data[num_test:-1]

batch_size = 100
num_epochs = 20
n_iters = int(num_epochs * len(data) / batch_size)

train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                          batch_size=batch_size, 
                                          shuffle=False)


'''
  Step 3: Make model
'''

class Brain(nn.Module):
  def __init__(self):
    super(Brain, self).__init__()

    # CNN 1
    self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=0)
    self.relu1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool2d(kernel_size=3)
    
    # CNN 2
    self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)
    self.relu2 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(kernel_size=3)

    # calculate size of output
    fake = torch.ones((1, 1, *shrink_size))
    # CNN1
    out = self.cnn1(fake)
    out = self.relu1(out)
    out = self.maxpool1(out)
    # CNN2
    out = self.cnn2(out)
    out = self.relu2(out)
    out = self.maxpool2(out)

    # 1 fc layer
    self.fc1 = nn.Linear(out.view(-1, 1).size()[0], 4)


  def forward(self, input):
    # CNN 1
    out = self.cnn1(input)
    out = self.relu1(out)
    out = self.maxpool1(out)

    # CNN 2
    out = self.cnn2(out)
    out = self.relu2(out)
    out = self.maxpool2(out)

    # FC
    out = out.view(out.size(0), -1)
    out = self.fc1(out)

    return out

'''
  Step 4: init model
'''

model = Brain()



'''
  Step 5: init loss function criteria
'''

criteria = nn.CrossEntropyLoss()

'''
  Step 6: init optimizer
'''

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


'''
  Step 7: train model
'''

iter = 0
for epoch in range(num_epochs):
  print("-------Epoch {}-------".format(epoch))
  for i, batch in enumerate(train_loader):
    iter += 1
    # print(iter)
    labels = batch['label']
    images = batch['img']

    images = Variable(torch.unsqueeze(images,1))
    labels = Variable(labels)

    # clear cumulated gradients
    optimizer.zero_grad()

    output = model(images)

    loss = criteria(output, labels)
    loss.backward()

    optimizer.step()

    if iter % 20 == 0:
      correct = 0
      total = 0

      # loop all test
      for i, batch in enumerate(test_loader):
        t_labels = batch['label']
        t_images = batch['img']

        t_images = Variable(torch.unsqueeze(t_images,1))
        
        t_out = model(t_images)

        _, precicted = torch.max(t_out, 1)

        total += t_labels.size(0)

        correct += (precicted == t_labels).sum().item()

      accuracy = correct / total
      print('Iteration: {}. Loss: {}. Accuracy: {:.2%}'.format(iter, loss.data.item(), accuracy))