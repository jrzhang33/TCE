
__all__ = ['ResBlock', 'ResNet']
from regulator.TCE import behavior

# Cell
from tsai.all import *
# from ..imports import *
# from .layers import *
# from .utils import *

# Cell
class ResBlock(Module):
    def __init__(self, ni, nf, kss=[8, 5, 3]):
        self.in_channel=ni
        self.out_channel=nf
        self.convblock1 = ConvBlock(ni, nf, kss[0])
        self.convblock2 = ConvBlock(nf, nf, kss[1])
        self.convblock3 = ConvBlock(nf, nf, kss[2], act=None)

        # expand channels for the sum if necessary
        self.shortcut = BN1d(ni) if ni == nf else ConvBlock(ni, nf, 1, act=None)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x
class Short(Module):
    def __init__(self, ni, nf):
        self.in_channel=ni
        self.out_channel=nf
        self.skip_shortcut =  ConvBlock(ni, nf, 1, act=None)
    def forward(self, x):
        x = self.skip_shortcut(x)
        return x
class ResNet(Module):
    def __init__(self, c_in, c_out,depth,skip,basic,regulate = False, verify = False):
        self.basic=basic
        nf = 64
        self.depth=depth
        kss=[8,5,3]
        self.skip=skip
        self.resblock,self.fc=self.res_layer(nf,kss,depth,c_in,c_out)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        
        self.x_out = []
        for i in range(depth+1):
            self.x_out.append(None)
        if verify == False:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                 #   nn.init.constant_(m.bias, 0)
                if  isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                if isinstance(m,nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                   nn.init.xavier_normal_(m.weight, gain=1.0)
                if isinstance(m,nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)   
    def short(self,in_channel,out_channel):
        self.short1=Short(in_channel,out_channel)
        return self.short1

    #Record p value for every layer to regulate
    def TCE(self):
        layer_p=[]
        for x in self.x_out:
            y_p=behavior(x)
            layer_p.append(y_p)
        return layer_p

    def res_layer(
        self,
        nf,
        kss,
        num_layer,
        c_in,
        c_out,
    ) -> nn.Sequential:
        BasicBlock = []
        RestBlock = []
        if self.depth<=2:
            basic_layer= self.depth
        else:
            basic_layer=2
        BasicBlock.append(
            ResBlock(c_in, nf, kss=kss))
        BasicBlock.append(ResBlock(nf, nf * 2, kss=kss))
        RestBlock.append(BasicBlock[0])
        for k in range(basic_layer, num_layer):
            layer = ResBlock(nf * 2, nf * 2, kss=kss)
            BasicBlock.append(layer)

        #Skip over distributing convolution in "skip" list
        for i in range(1,num_layer): 
            if i in self.skip:
                continue
            #If the dimensions are inconsistent between the input channel and input, use shortcuts
            else:
                if RestBlock[-1].out_channel!=BasicBlock[i].in_channel:
                        RestBlock.append(Short(RestBlock[-1].out_channel,BasicBlock[i].in_channel) )
                RestBlock.append(BasicBlock[i])        

            
        self.fc = nn.Linear(RestBlock[-1].out_channel, c_out)
        
        return nn.Sequential(*RestBlock),self.fc

    def forward(self, x):
        x0=self.resblock[0].convblock1(x)
        self.x_out[0]=x0  
        if self.basic:
            for i in range(0,len(self.resblock)):
                x= self.resblock[i](x) 
                self.x_out[i+1]=x
        else:
            for i in range(0,len(self.resblock)):
                x= self.resblock[i](x) 

        x = self.squeeze(self.gap(x))
        return self.fc(x)