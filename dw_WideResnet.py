
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Conv_ReLU_Block(nn.Module):
	def __init__(self):
		super(Conv_ReLU_Block, self).__init__()
		self.conv = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self, x):
		return self.relu(self.conv(x))

class BasicBlock(nn.Module):

    def __init__(self, inplanes=32, planes=32):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class wBasicBlock(nn.Module):

	def __init__(self, inplanes=32, planes=32, dropout_rate=0.5):
		super(wBasicBlock, self).__init__()
		self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
		self.bn1 = nn.BatchNorm3d(planes)
		self.relu = nn.ReLU(inplace=True)
	
		self.dropout = nn.Dropout3d(p=dropout_rate)
	
		self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
		self.bn2 = nn.BatchNorm3d(planes)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.dropout(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out += residual
		out = self.relu(out)

		return out

#%%
class WideResNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.lambda_val = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad = True)
		self.p=torch.nn.Parameter(torch.Tensor([2]), requires_grad = True)
		self.gen = nn.Sequential(
				nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
				nn.ReLU(inplace=True),
				self.make_layer(wBasicBlock, 8),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
		)
				
	def make_layer(self, block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block())
		return nn.Sequential(*layers)

	def forward(self,x_input):
		x_pred =self.gen(x_input)
		return x_pred


#%%

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        #print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("Toc: start time not set")


#%%
# device_id=1


# net=WideResNet();
# net=net.cuda(device_id)
# print(net)

# x=torch.randn(1,2,176,176,160)
# x=x.cuda(device_id);
# tic()
# with torch.no_grad():
#     y=net(x)
# toc()
# print(y.shape)





# #%%
# for i in net.parameters():
#     print(i.shape,i.name)
    
# #%%
# pp=0
# for p in (net.parameters()):
#     nn=1
#     for s in p.size():
#         nn=nn*s
#     pp+=nn      
# print('number of parameters: ',pp)


#%%
# import torch
# from torchvision import models
# from torchsummary import summary


# summary(net, (176,160,160))



