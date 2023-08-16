from regulator.TCE import behavior

# Cell
from tsai.all import *

class Short(Module):
    def __init__(self, ni, nf):
        self.in_channel=ni
        self.out_channel=nf
        self.skip_shortcut =  ConvBlock(ni, nf, 1)
    def forward(self, x):
        x = self.skip_shortcut(x)
        return x

class FCNLayer(Module):
    def __init__(self, ni, nf, kss):
        self.in_channel=ni
        self.out_channel=nf
        self.convblock = ConvBlock(ni, nf, kss)


    def forward(self, x):
        x = self.convblock(x)
        return x

class FCN(Module):
    '''FCN'''
    def __init__(self, c_in, c_out,depth,skip,basic,regulate = False):
        
        self.basic=basic
        nf = 128
        self.depth=depth
        kss = [8,5,3,8,5,3,3,3]
        self.skip=skip
        self.fcnlayer,self.fc=self.FCN_layer(nf,kss,depth,c_in,c_out)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.x_out = []
        for i in range(depth+1):
            self.x_out.append(None)
        if regulate:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                if  isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                if isinstance(m,nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        else:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                   # kaiming_normal_(m.weight)
                if  isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                if isinstance(m,nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def short(self,in_channel,out_channel):
        self.short1=Short(in_channel,out_channel)
        return self.short1
    def TCE(self):
        layer_p=[]
        for x in self.x_out:
            y_p=behavior(x)
            layer_p.append(y_p)
        return layer_p
    
    def FCN_layer(
        self,
        nf,
        kss,
        num_layer,
        c_in,
        c_out,
    ) -> nn.Sequential:
        BasicLayer = []
        RestLayer = []
        if self.depth <= 2:
            basic_layer= self.depth
        else:
            basic_layer=2

        BasicLayer.append(
            FCNLayer(c_in, nf, kss[0]))
        BasicLayer.append(FCNLayer(nf, nf * 2, kss[1]))
        RestLayer.append(BasicLayer[0])

        # BasicLayer.append(FCNLayer(nf*2, nf * 4, kss[2]))
        # BasicLayer.append(FCNLayer(nf*4, nf * 2, kss[3]))
        # BasicLayer.append(FCNLayer(nf*2, nf , kss[4]))
        


        for k in range(basic_layer, num_layer - 1):
            layer = FCNLayer(nf * 2, nf * 2, kss[k])
            BasicLayer.append(layer)
        layer = FCNLayer(nf * 2, nf , kss[-1])
        BasicLayer.append(layer)
        #Skip over distributing convolution in "skip" list
        for i in range(1,num_layer): 
            if i in self.skip:
                continue
            #If the dimensions are inconsistent between the input channel and input, use shortcuts
            else:
                if RestLayer[-1].out_channel!=BasicLayer[i].in_channel:
                        RestLayer.append(Short(RestLayer[-1].out_channel,BasicLayer[i].in_channel) )
                RestLayer.append(BasicLayer[i])        

            
        fc = nn.Linear(RestLayer[-1].out_channel, c_out)
        
        return nn.Sequential(*RestLayer),fc


    def forward(self, x):
        x0=self.fcnlayer[0].convblock(x)
        self.x_out[0]=x0  
        if self.basic:
            for i in range(0,len(self.fcnlayer)):
                x= self.fcnlayer[i](x) 
                self.x_out[i+1] = x
                

                #self.x_out[i+1]=x.copy()
        else:
            for i in range(0,len(self.fcnlayer)):
                x= self.fcnlayer[i](x)

        x = self.squeeze(self.gap(x))
        return self.fc(x)

# if __name__ == "__main__":
#     model = FCN(5, 3)
#     x = torch.randn(64, 5, 96) # bs, chs, seq_len
#     y = model(x)
#     print(y.size())
