import torch
import torch.nn.functional as F
import torch.nn

class Linear(torch.nn.Module):
    def __init__(self,int,out):
        super().__init__()

        self.mlp=torch.nn.Linear(int, out)
    def forward(self, x):
        out=self.mlp(x)
        return out

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu1=torch.nn.ReLU()
        self.max_pooling1=torch.nn.MaxPool2d(2,1)
 
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu2=torch.nn.ReLU()
        self.max_pooling2=torch.nn.MaxPool2d(2,1)
 
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)
        self.dense3 = Linear(10,5)

 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pooling2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
 
model = MyNet() # 构造模型

for name in model.state_dict().keys(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    print(name,type(name))
for values in model.state_dict().values(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    print(values.shape)
    
for key,value in model.state_dict().items(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    print(key,value.shape)

'''
print(type(model.parameters()))  # 返回的是一个generator
 
for para in model.parameters():
    print(para.size())  # 只查看形状


print(type(model.named_parameters()))  # 返回的是一个generator
 

 '''