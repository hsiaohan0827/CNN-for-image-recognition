# CNN-for-image-recognition
implement a simple CNN model using pytorch

## Setup
1. 創建一個新環境
```
python3 -m venv env_name
```
2. activate environment
```
source env_name/bin/activate
```
3. 安裝requirement.txt中的套件
```
pip3 install -r requirements.txt
```


## Download Data
1. Using Medical Masks Dataset (https://www.kaggle.com/vtech6/medical-masks-dataset), comes from Eden Social Welfare Foundation which contains the pictures of people wearing medical masks along with the labels containing their descriptions

2. Download images and labels, transforming .xml to a .csv file, with header row 'filename', 'label', 'xmax', 'xmin', 'ymax', 'ymin'.
   For example:
   | filename | label | xmax | xmin | ymax | ymin |
   | -------- | :---: | :--: | :--: | :--: | :--: |
   |c1\_1844849.jpg|good|1246|127|1312|227|
   |c1\_1844849.jpg|none|745|889|862|999|
   
3. Split data for train and test, name the file as train.csv / test.csv.



## Training
1.  run preprocess.py for both train.csv and test.csv.  
   (remember to edit different output filename: croppedImg_xxxx / imgLabel_xxxx)
```
python3 preprocess.py
```

2.  修改cnn.py中的config
```python
# CONFIG
output_dir = 'AlexNet_is32_bs256_ep300_loss'
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
```

### configuration
- **output_dir** - model name.
- **epochs** - epoch number.
- **bch_size** - batch size.
- **lr** - learning rate.
- **imgSize** - input img size.
- **save_freq** - save modal every save_freq epochs
- **istrain** - if the model do train or test
- **modelPath** - model path for testing, work if istrain=False

3.  run cnn.py
```
python3 cnn.py
```

### tensorboardX
可以使用tensorboard觀察loss及accuracy變化
```
tensorboard --logdir CNN_log
```

## Testing
1. 修改istrain及modelPath
```python
istrain = False
modelPath = 'AlexNet_is32_bs256_ep300_loss/ep250.pkl'
```
2. run cnn.py, get classification result for each class.
```
python3 cnn.py
```

3. 可以在main中更換不同optimizer
```python
if __name__ == '__main__':

    ......

    if istrain:
        # create model & optimizer
        cnn = CNN()
        cnn.apply(initialize_parameters)
        print(cnn) 
        
        # CHOOSE AN OPTIMIZER
        optimizer = torch.optim.Adam(cnn.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08)
        #optimizer = torch.optim.SGD(cnn.parameters(),lr=lr)
        #optimizer = torch.optim.SGD(cnn.parameters(),lr=lr,momentum=0.8)        
        #optimizer = torch.optim.RMSprop(cnn.parameters(),lr=lr,alpha=0.9)

    ......
```
