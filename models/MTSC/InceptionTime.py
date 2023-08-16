__all__ = ['InceptionModule', 'InceptionBlock', 'InceptionTime']

from tsai.all import *
from regulator.TCE import behavior
class InceptionModule(Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True,regulate = False):
        # ks = [ks // (2**i) for i in range(3)]
        # ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        ks = [10, 20, 40]
        bottleneck = bottleneck if ni > 1 else False
        self.bottleneck = Conv1d(ni, nf, 1, bias=False) if bottleneck else noop
        self.convs = nn.ModuleList([Conv1d(nf if bottleneck else ni, nf, k, bias=False) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), Conv1d(ni, nf, 1, bias=False)])
        self.concat = Concat()
        self.bn = BN1d(nf * 4)
        self.act = nn.ReLU()
        if regulate:
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                 #   nn.init.constant_(m.bias, 0)
                if isinstance(m,nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return self.act(self.bn(x))


@delegates(InceptionModule.__init__)
#"basic" means choose or not choose TCE (True or False)
class InceptionBlock(Module):
    def __init__(self, ni,depth,skip,basic, nf=32,residual=True,**kwargs):
        self.residual, self.depth, self.skip = residual, depth, skip
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        self.inceptionblock=[]
        dd=0
        for d in range(depth):
            self.inceptionblock.append(InceptionModule(ni if d == 0 else nf * 4, nf, **kwargs))
        #Skip distributing convolution in "skip" list
        for d in range(depth):
            if d in self.skip:
                continue
            else:
                self.inception.append(self.inceptionblock[d])
                if self.residual and dd % 3 == 2:
                    n_in, n_out = ni if dd == 2 else nf * 4, nf * 4
                    self.shortcut.append(BN1d(n_in) if n_in == n_out else ConvBlock(n_in, n_out, 1, act=None))
                dd=dd+1
        self.add = Add()
        self.act = nn.ReLU()
        #Record the output of every block, the first and the second value in list are the first output
        self.x_out = []  
        self.basic=basic
        for i in range(depth+1):
            self.x_out.append(None)
         
    def forward(self, x):
        res = x
        for d, l in enumerate(range(len(self.inception))):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2: res = x = self.act(self.add(x, self.shortcut[d//3](res)))
            # save output
            if self.basic:
                if d==0:
                    self.x_out[0]=x   
                self.x_out[d+1]=x
        return x


@delegates(InceptionModule.__init__)
class InceptionTime(Module):
    def __init__(self, c_in, c_out, depth,skip,basic=False, seq_len=None, nf=32,nb_filters=None, **kwargs):
        nf = ifnone(nf, nb_filters) # for compatibility
        self.inceptionblock = InceptionBlock(c_in, depth,skip,basic,nf, **kwargs)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(nf * 4, c_out)
    #Record p value for every layer to regulate
    def TCE(self):
        layer_p=[]
        for x in self.inceptionblock.x_out:
            y_p=behavior(x)
            layer_p.append(y_p)
        return layer_p
    def forward(self, x):
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.fc(x)
        return x