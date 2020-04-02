import torch.nn as nn
import torch
import  numpy as np
import torchvision
from torchvision import datasets, models,transforms
import  os ,re ,glob
import torch.optim as optim
import h5py

data_dir = "/data/dataset/ILSVRC2012"
save_dir = 'checkpoint/'
model_name = "resnet"
num_classes = 1000
batch_size = 50
num_epochs = 20

# load pretrained model
model = models.resnet50(pretrained=True)
print("model structure of resnet-50")
print (model)

# load data
# train_loader = get_imagenet_loader(batch_size=50, num_workers=4, h5_path="data/train_imagenet_50000.h5",shuffle=True)
# test_loader = get_imagenet_loader(batch_size=50, num_workers=4, h5_path="data/test_imagenet_50000.h5",shuffle=False)

# data_transforms = transforms.Compose([transforms.RandomSizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
data_transforms = transforms.Compose([transforms.RandomSizedCrop(64),transforms.ToTensor()])
print("Initializing Datasets and DataLoaders...")
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms) for x in ['train','val']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True,num_workers=4) for x in ['train','val']}

samples = {"train":100000,'val':50000}

# load device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# define optimizer and loss func
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
criterion = nn.CrossEntropyLoss()

# init_epoch = 7
# model = torch.load("checkpoint/resnet50_epoch_7.pth")
# model = model.to(device)


for epoch in range(num_epochs):
    print ('Epoch {}/{}'.format(epoch,num_epochs-1))
    print ('-' * 10)

    for phase in ['train','val']:
        if phase == "train":
            continue
        model.train() if phase == 'train' else model.eval()
        running_loss = 0.0
        running_corrects = 0

        # hash = torch.zeros([1000,1000])

        for batch_idx ,[data,target] in enumerate(dataloaders_dict[phase]):
            print("{}/{}".format(batch_idx,samples[phase]/batch_size))
            optimizer.zero_grad()
            data,target = data.to(device),target.to(device)
            if batch_idx*batch_size >= samples[phase]:
                break

            if phase == 'train':
                output = model(data)
            else:
                with torch.no_grad():
                    output = model(data)

            pred =   output.max(1,keepdim=True)[1]
            # for i in range(pred.size()[0]):
            #     hash[batch_idx,pred[i]] += 1

            running_corrects += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output,target)
            running_loss += loss.item()
            if phase == 'train':
                loss.backward()
                optimizer.step()
        print('{} loss: {:.4f} Acc: {}/{} ({:.4f})'.format(phase,running_loss,running_corrects,samples[phase],running_corrects/samples[phase]))

        # hash = hash.max(1,keepdim=True)[1]
        # print(hash)

    torch.save(model,os.path.join('checkpoint',"64_resnet50_epoch_"+str(epoch)+".pth"))


