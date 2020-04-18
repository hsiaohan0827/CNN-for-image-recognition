import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import csv
from tensorboardX import SummaryWriter
import cv2
from torchvision import transforms


# config
output_dir = 'AlexNet_is32_bs256_ep500_RMSprop'
if not os.path.isdir('CNN_model/'+output_dir):
    os.mkdir('CNN_model/'+output_dir)
logger = SummaryWriter('CNN_log/'+output_dir)

epochs = 500
bch_size = 256
lr = 0.001
imgSize = 32
save_freq = 50
istrain = True
modelPath = 'AlexNet_is32_bs256_ep300_loss+w/ep250.pkl'


# create dataset
class TorchDataset(Data.Dataset):
    def __init__(self, trainData):
        if trainData:
            self.labelPath = 'imgLabel_train.npy'
            self.imgPath = 'croppedImg_train'
            self.transforms = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = (97.15083268 / 255., 104.22849715 / 255., 120.50918226 / 255.), 
                                                 std = (62.48380569 / 255., 61.05203275 / 255., 63.17664679 / 255.))
                       ])
        else:
            self.labelPath = 'imgLabel_test.npy'
            self.imgPath = 'croppedImg_test'
            self.transforms = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = (93.4769411 / 255., 100.39986516 / 255., 115.63268299 / 255.), 
                                                 std = (63.2985143 / 255., 61.99833393 / 255., 63.92628052 / 255.))
                       ])

        self.label_list = np.load(self.labelPath)
        self.len = len(self.label_list)

        
    def __getitem__(self, i):
        label = np.array(self.label_list[i])
        img = cv2.imread(os.path.join(self.imgPath, str(i)+'.jpg'))
        img = self.transforms(img)
        #img.type(torch.FloatTensor) / 255.
        return img, label

    def __len__(self):
        return len(self.label_list)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
                #in_channels, out_channels, kernel_size, stride, padding
                nn.Conv2d(3, 64, 3, 2, 1),             # [3 x 32 x 32] -> [64 x 16 x 16]
                nn.MaxPool2d(2),                       # [64 x 16 x 16] -> [64 x 8 x 8]
                nn.ReLU(inplace = True),
                nn.Conv2d(64, 192, 3, 1, padding = 1),    # [64 x 8 x 8] -> [192 x 8 x 8]
                nn.MaxPool2d(2),                       # [192 x 8 x 8] -> [192 x 4 x 4]
                nn.ReLU(inplace = True),
                nn.Conv2d(192, 384, 3, 1, padding = 1),   # [192 x 4 x 4] -> [384 x 4 x 4]
                nn.ReLU(inplace = True),
                nn.Conv2d(384, 256, 3, 1, padding = 1),   # [384 x 4 x 4] -> [256 x 4 x 4]
                nn.ReLU(inplace = True),
                nn.Conv2d(256, 256, 3, 1, padding = 1),   # [256 x 4 x 4] -> [256 x 4 x 4]
                nn.MaxPool2d(2),                       # [256 x 4 x 4] -> [256 x 2 x 2]
                nn.ReLU(inplace = True)
            )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096),              # [256 x 4 x 4] -> [4096]
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),                     # [4096] -> [4096]
            nn.ReLU(inplace = True),
            nn.Linear(4096, 3),                        # [4096] -> [3]
        )
    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    if istrain:
        acc = correct.float() / y.shape[0]
    else:
        acc = correct.float()
    return acc

def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)


def train(cnn, optimizer, criterion, train_loader, test_loader):
    for epoch in range(epochs):
        train_acc = 0.
        train_loss = 0.

        # start training
        for step, (img, label) in enumerate(train_loader):
            
            img = img.to(device)
            label = label.to(device)
            cnn.train()
            output = cnn(img)[0]
            one_hot = torch.max(label, 1)[1]
            loss = criterion(output, one_hot)
            train_loss += loss.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # training accuracy
            train_acc += calculate_accuracy(output, one_hot)

        cnn.eval()  
        test_loss = 0.
        test_acc= 0.

        # starting testing
        for i, (img, label) in enumerate(test_loader):
            img = img.to(device)
            label = label.to(device)
            output = cnn(img)[0]
            one_hot = torch.max(label, 1)[1]
            loss = criterion(output, one_hot)
            test_loss += loss.data
            
            # testing accuracy
            test_acc += calculate_accuracy(output, one_hot)

        # save model
        if epoch % save_freq == 0:
            torch.save(cnn.state_dict(), os.path.join('CNN_model', output_dir, 'ep'+str(epoch)+'.pkl'))


        print('epoch: %d test loss: %f train loss: %f test acc: %f train acc: %f'
                    %(epoch, test_loss / len(test_loader), train_loss / len(train_loader),
                            test_acc / len(test_loader), train_acc / len(train_loader)))
        logger.add_scalar('accracy/train', train_acc / len(train_loader), epoch)
        logger.add_scalar('accracy/test', test_acc / len(test_loader), epoch)
        logger.add_scalar('loss/train', train_loss / len(train_loader), epoch)
        logger.add_scalar('loss/test', test_loss / len(test_loader), epoch)


def predict(cnn, criterion, train_loader, test_loader):
    cnn.eval()  
    test_good = 0.
    test_bad = 0.
    test_none = 0.
    train_good = 0.
    train_bad = 0.
    train_none = 0.

    # starting testing
    for i, (img, label) in enumerate(test_loader):
        img = img.to(device)
        label = label.to(device)
        output = cnn(img)[0]
        one_hot = torch.max(label, 1)[1]

        good = one_hot[one_hot==0]
        bad = one_hot[one_hot==1]
        none = one_hot[one_hot==2]
        test_predict_good = output[one_hot==0]
        test_predict_bad = output[one_hot==1]
        test_predict_none = output[one_hot==2]

        # testing accuracy
        test_good += calculate_accuracy(test_predict_good, good)
        test_bad += calculate_accuracy(test_predict_bad, bad)
        test_none += calculate_accuracy(test_predict_none, none)

    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        output = cnn(img)[0]
        one_hot = torch.max(label, 1)[1]

        good = one_hot[one_hot==0]
        bad = one_hot[one_hot==1]
        none = one_hot[one_hot==2]
        train_predict_good = output[one_hot==0]
        train_predict_bad = output[one_hot==1]
        train_predict_none = output[one_hot==2]


        # testing train_predict_good
        train_good += calculate_accuracy(train_predict_good, good)
        train_bad += calculate_accuracy(train_predict_bad, bad)
        train_none += calculate_accuracy(train_predict_none, none)

    print('test: good - %f  bad - %f  none - %f'%(test_good / 283., test_bad / 89., test_none / 22.))
    print('train: good - %f  bad - %f  none - %f'%(train_good / 2846., train_bad / 578., train_none / 104.))



if __name__ == '__main__':

    # create dataloader
    train_data = TorchDataset(trainData=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=bch_size, shuffle=True)

    test_data = TorchDataset(trainData=False)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=bch_size, shuffle=False)

    print('train data: '+str(len(train_data)))
    print('test data: '+str(len(test_data)))

    # define loss function
    #weights = [1./3129, 1./667, 1./126]
    weights = [1, 1, 1]
    class_weights = torch.FloatTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)


    if istrain:
        # create model & optimizer
        cnn = CNN()
        cnn.apply(initialize_parameters)
        print(cnn) 
        optimizer = torch.optim.Adam(cnn.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08)
        #optimizer = torch.optim.SGD(cnn.parameters(),lr=lr)
        #optimizer = torch.optim.SGD(cnn.parameters(),lr=lr,momentum=0.8)        
        #optimizer = torch.optim.RMSprop(cnn.parameters(),lr=lr,alpha=0.9)

        device = torch.device("cuda")
        cnn = cnn.to(device)
        criterion = criterion.to(device)

        train(cnn, optimizer, criterion, train_loader, test_loader)
    else:
        # load model
        cnn = CNN()
        cnn.load_state_dict(torch.load(os.path.join('CNN_model', modelPath)))
        
        device = torch.device("cuda")
        cnn = cnn.to(device)
        criterion = criterion.to(device)

        predict(cnn, criterion, train_loader, test_loader)

    