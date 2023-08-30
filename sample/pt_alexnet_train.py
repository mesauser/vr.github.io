import torch
import torchvision
from torchvision.models import resnet18
from torch import nn
# model = resnet18(num_classes=1000) # MNIST has 10 classes
# layer = nn.Linear(1000,10)

import torchvision.transforms as transforms
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((1.0), (1.0))
])

ds = torchvision.datasets.mnist.MNIST("mnist", train=True, download=True, transform=trans)
ds_load = torch.utils.data.DataLoader(dataset=ds, batch_size=100, shuffle=True)

class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = torchvision.models.alexnet(num_classes=10)
#         self.alexnet = torchvision.models.alexnet(pretrained=True).cuda()
        self.upsample = torch.nn.Upsample(scale_factor=8, mode="nearest") # 28x28 -> 224x224
        # upsample = torch.nn.Upsample(size=(4,4), mode="nearest")
        self.conv1 = torch.nn.Conv2d(1, 3, (1,1))       # C=1 -> C=3
        self.fc = torch.nn.Linear(1000,10)
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, data):
#         ux = self.upsample(data)
#         cx = self.conv1(ux)
        o= self.alexnet(data)
        return o

mynet = MyNet()
optimizer = torch.optim.SGD(mynet.parameters(), lr=0.1, momentum=0.9)
optimizer.zero_grad()
loss_fun=torch.nn.CrossEntropyLoss()

# mynet = mynet.cuda()
# optimizer = optimizer.cuda()
# loss_func = loss_func.cuda()

for idx, (x,y) in enumerate(ds_load):
    data = x.repeat(1,3,1,1)
#     data = data.cuda()
#     y=y.cuda()
    a=mynet(data)
    loss = loss_fun(a, y)
    loss.backward()
    optimizer.step()
    print(loss)
    
    if idx % 100 == 0:
        print(f"{idx} loss={loss}")

