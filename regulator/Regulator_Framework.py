import torch
import torch.utils.data
import torch
def Regulator(model):
    min_skip=2 # The maximum number of layers that can be skipped
    layer_p=model.TCE() # Obtain the list of p for every layer (Eq.2) 
    num=1
    var=[]  # the list of p for all FE feature maps in corresponding layer (Eq.3) 
    layer_M=[]  # the list of M (Remark 1) for layer from the second layer
    skip=[]  # skip list, number start from 1 and layer skipped over from the second layer.
    # Go through the p at each layer, and we compute the v (Eq.3) at each layer
    focus_scale = []
    for i in layer_p[1:]:  
        y_p=i 
        y_p=torch.where(torch.isnan(y_p), torch.full_like(y_p, 0), y_p)    
        _k = torch.var(y_p, dim=1,unbiased=False)  # Obtain v (Eq.3)
        var_k=torch.mean(_k,dim=0) # Get the average of all samples of v
        if num==1:  
            last_var=var_k.squeeze().cpu().item() #the first layer
            focus_scale.append(last_var)
        else:
            M=var_k.squeeze().cpu().item()-last_var # M, the difference v between layer and its previous layer (Remark 1)
            last_var=var_k.squeeze().cpu().item()
            layer_M.append(M) 
            var.append(last_var)
            focus_scale.append(last_var)
        num=num+1
    # Find distributing convolution
    while True:
        if len(layer_M )== 0 or len(layer_M) == 1: 
            return []      
        x = min(layer_M)
        # distributing convolution
        if x < 0 and len(skip) < min_skip: 
            find = (layer_M.index(x))
            skip.append(find+1) 
            layer_M[find]=0
        else:
            return skip, focus_scale

 

    
