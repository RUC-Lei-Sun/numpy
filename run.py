
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import f1_score
import collections


x=torch.arange(12).reshape(2,2,3)
x=x.float()

lstm=nn.LSTM(3,3,batch_first=True)
output,(ht,ct)=lstm(x)
print(output[:,-1,:])
print(output[:,-1,:].shape)
print(ht)
print(ht.shape)

x=torch.arange(20).reshape(4,5)
index=[1,3]
#index=torch.tensor([1,3])
print(x[0])          # 输出最后一行数据
print(x[1:3])         # 输出第2行~第3行全部数据（从1算起，3前为止）
print(x[index])







'''
net=nn.Softmax(dim=0)
print(x)
print(net(x))
'''
'''
e=torch.tensor(-1)
print(e)

a = torch.rand((2, 3))

b=torch.tensor([1,2])
print(b)

'''
'''
y_true = [0, 2, 2, 2, 2, 2]
y_pred = [0, 2, 2, 2, 2, 2]

y_true=np.array(y_true)
y_pred=np.array(y_pred)

y_pred=y_pred[y_true!=0]
y_true =y_true[y_true!=0]


print("y_true",y_true)
print("y_pred",y_pred)

f1 = f1_score(y_true, y_pred, average=None)
print(f1)

'''
'''
image=Image.open("JSXL.jpg")
transforms=transforms.Compose([
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
image=transforms(image)
print(image.shape)
'''
'''
f = h5py.File("myh5py.h5", "w")

# 创建组bar1,组bar2，数据集dset
g1 = f.create_group("bar1")
g2 = f.create_group("bar2")
d = f.create_dataset("dset", data=np.arange(10))

# 在bar1组里面创建一个组car1和一个数据集dset1。
c1 = g1.create_group("car1")
d1 = g1.create_dataset("dset1", data=np.arange(10))

# 在bar2组里面创建一个组car2和一个数据集dset2
c2 = g2.create_group("car2")
d2 = g2.create_dataset("dset2", data=np.arange(10))

# 根目录下的组和数据集
print(".............")
for key in f.keys():
    print(f[key].name)


# bar1这个组下面的组和数据集
print(".............")
for key in g1.keys():
    print(g1[key].name)

# bar2这个组下面的组和数据集
print(".............")
for key in g2.keys():
    print(g2[key].name)

# 查看car1组和car2组下面都有什么
print(".............")
print(c1.keys())
print(c2.keys())
'''