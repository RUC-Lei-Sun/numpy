import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp1=nn.Sequential(
            nn.Linear(5,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Dropout()
        )
       

    def forward(self,batch_data):
        out=self.mlp1(batch_data)
        return out



if __name__=="__main__":
    x=torch.ones(4,5)

    model=Test()
    out=model(x)
    print(out)
    print(out.shape)
    for name, param in model.named_parameters():
        print(name,type(name), param.size(), type(param))

