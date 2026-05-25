# neural-network-project1-comparison-between-mlp-and-cnn
## cnn structure
myCNN(

  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (dropout): Dropout(p=0.25, inplace=False)
  
  (relu): ReLU()
  
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  
)
## cnn performance on mnist without data augmentation
train_acc = 1.0
val_acc = 0.9878
test_acc = 0.9842
