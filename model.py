import torch
import torch.nn as nn
import swin_transformer
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class WiSwin(nn.Module):
    def __init__(self):
        super(WiSwin,self).__init__()
        self.swin=swin_transformer.SwinTransformer(
            img_size=20,patch_size=2,in_chans=1,
            embed_dim=64,depths=[2,2],num_heads=[2,4],
            window_size=5)
        #self.layer_norm = nn.LayerNorm([20,20])
       

        self.mlp=nn.Sequential(
            nn.Linear(128,128),
            nn.GELU(),
            nn.Linear(128,128)
        )

        self.lstm=nn.LSTM(input_size=128,hidden_size=128,num_layers=1,bidirectional=True)
        self.dropout=nn.Dropout(0.2)
        self.fc=nn.Linear(256,6)

    def forward(self,x,lengths):
        #B T 20 20
        # Time_seq=x.shape[1]
        # x_dot=[]
        B, T, H, W = x.shape
        """
        if self.layer_norm.weight.device != x.device:
            self.layer_norm = self.layer_norm.to(x.device)
        x = self.layer_norm(x).view(B*T,1,H,W)
        """
        x=x.view(B*T,1,H,W)
        

        x = self.swin(x)  # 最快路径：无随机深度，全部 tensor core

        #x=self.mlp(x)  #B*T 128
        
        x=x.view(B,T,x.size(-1)) #B T 128
        packed=pack_padded_sequence(
            x,lengths.cpu(),batch_first=True,enforce_sorted=True
        )
        _,(ht,ct)=self.lstm(packed)
        ht=torch.cat((ht[0],ht[1]),dim=-1)
        ht = self.dropout(ht) # 缓解过拟合
        outputs=self.fc(ht)
        """
        packed_outputs,_=self.lstm(packed)
        out,_=pad_packed_sequence(packed_outputs,batch_first=True)
        feat=torch.mean(out,dim=1)
        feat=self.dropout(feat)
        outputs=self.fc(feat)
        """
        return outputs