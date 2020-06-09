import sys
sys.path.insert(0, './')
import torch
from thop import profile
from thop import clever_format
from patchnet import PatchNetv3g, PatchNetv3
from patchnet.fourier_feature import FourierFeature
from patchnet.pool_select import MaxPoolSoftSelect
from patchnet.op_count import count_fourierfeat, count_poolsoftselect

def main():
    patchnet = PatchNetv3()   
    patchnet.create_architecture()                     
    patchnet.eval()                                    
    patchnet.cuda()                                    

    x = torch.randn(5,3,119,119).cuda()
    z = torch.randn(5,3,64,64).cuda()
    flops, params = profile(
            patchnet, 
            inputs=(x, z),
            custom_ops={FourierFeature: count_fourierfeat,
                        MaxPoolSoftSelect: count_poolsoftselect,
                        })
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

if __name__ == "__main__":
    main()
