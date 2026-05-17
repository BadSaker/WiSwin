import scipy.io as scio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
import os
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
#root_dir='/mnt/e/BaiduNetdiskDownload/Widar3.0ReleaseData/BVP/gesture_groups'
root_dir='C:/Users/林鹏/Desktop/Files/WiSwin/BVP/gesture_groups'

class WidarGestureDataset(Dataset):
    def __init__(self,path,setting):
        self.bvp=[]
        self.ges=[]
        self.domain_labels=[]
        self.files=[self.get_filelist(path)]
        label_map = get_label_mapping(setting)
    
        for ff in self.files:
            for f in ff:
                f = f.replace('\\', '/')
                ges_type=f.split('/')[-2]
                if ges_type in label_map:
                    filename=f.split('/')[-1]
                    parts = filename.replace('.mat','').split('-')
                    user_id=int(''.join(filter(str.isdigit, parts[0])))
                    location_id=int(parts[2])
                    room_id=int(parts[5])

                    # if setting==1 and user_id>5:
                    #     continue
                    # if setting==2 and user_id>4:
                    #     continue
                    
                    # if location_id>5:
                    #     continue
                    try:
                        mat = scio.loadmat(f)
                    except Exception as e:
                        # 打印出坏文件的路径，方便你以后手动删掉它，然后直接跳过
                        print(f"⚠️ 警告：文件损坏或被截断，已跳过 -> {f}")
                        continue
                    
                    
                    velocity = mat['velocity_spectrum_ro']  # shape: (H, W, T)

                    if len(velocity.shape) == 3 and velocity.shape[2] != 0:
                        data=torch.from_numpy(mat['velocity_spectrum_ro']).float()
                        data=normalize_data(data)
                        self.bvp.append(data.permute(2,0,1))  # 转换为 (T, H, W)
                        self.ges.append(label_map[ges_type])

                        if setting==2:
                            # 存储一个字典，包含该样本所有的域信息
                            self.domain_labels.append({
                            'user': user_id,
                            'location': location_id,
                            'room': room_id
                            })

        self.ges=torch.LongTensor(self.ges)

    def __len__(self):
        return len(self.bvp)
    
    def __getitem__(self, idx):
        return self.bvp[idx],self.ges[idx]

    def get_domain_label(self,idx):
        return self.domain_labels[idx]
    
    def get_filelist(self,path):
        file_list=[]
        for root,dir,files in os.walk(path):
            for file in files:
                file_list.append(os.path.join(root,file))
        return file_list
    
def normalize_data(data):
    """
    data: torch.Tensor, shape (H, W, T)
    return: normalized tensor (H, W, target_T)
    """
    # H, W, T = data.shape
    # target_T = 20  # 论文中标准化到 20 帧
    
    # # 防止空数据
    # if T == 0:
    #     return data

    # # === 1. 时间序列实例归一化 (论文 5.1 节逻辑) ===
    # if T != target_T:
    #     # 计算缩放因子 t / t0
    #     scale = T / target_T
        
    #     # (1) 速度坐标系缩放 (Spatial Zoom)
    #     # 动作较慢(T > target_T)时，scale > 1，需将图像向外围放大，代表速度提升
    #     # PyTorch 的 affine_grid 需要传入变换矩阵的逆。缩放图像即对角线设为 1/scale
    #     theta = torch.tensor([
    #         [1.0 / scale, 0.0, 0.0],
    #         [0.0, 1.0 / scale, 0.0]
    #     ], dtype=torch.float32, device=data.device)
        
    #     # 将矩阵扩展以应用于所有 T 帧
    #     theta = theta.repeat(T, 1, 1)
        
    #     # 生成采样网格，大小为 (T, 1, H, W)
    #     grid = F.affine_grid(theta, size=(T, 1, H, W), align_corners=False)
        
    #     # 调整数据形状为 (T, C, H, W) 以适配 grid_sample
    #     data_swapped = data.permute(2, 0, 1).unsqueeze(1) # (T, 1, H, W)
        
    #     # 执行速度平面的空间插值缩放
    #     data_scaled = F.grid_sample(data_swapped, grid, align_corners=False, padding_mode='zeros')
        
    #     # (2) 时间维度重采样 (Temporal Resampling)
    #     # 调整形状为 (Batch=1, Channels=H*W, Length=T) 以便进行 1D 插值
    #     data_scaled = data_scaled.squeeze(1).view(T, -1).permute(1, 0).unsqueeze(0) 
        
    #     # 线性插值到标准长度 target_T
    #     data_resampled = F.interpolate(data_scaled, size=target_T, mode='linear', align_corners=False)
        
    #     # 恢复回 (H, W, target_T) 的形状
    #     data = data_resampled.squeeze(0).view(H, W, target_T)

    # 按官方逻辑计算 max
    max_axis0 = torch.max(data, dim=0).values   # (W, T)
    max_axis1 = torch.max(data, dim=1).values   # (H, T)
    data_max = torch.max(torch.cat([max_axis0, max_axis1], dim=0), dim=0).values  # (T,)

    # 计算 min
    min_axis0 = torch.min(data, dim=0).values
    min_axis1 = torch.min(data, dim=1).values
    data_min = torch.min(torch.cat([min_axis0, min_axis1], dim=0), dim=0).values  # (T,)

    # 防止除零
    diff = data_max - data_min
    if torch.any(diff == 0):
        return data

    # 扩展维度 -> (1,1,T)
    data_max = data_max.unsqueeze(0).unsqueeze(0)
    data_min = data_min.unsqueeze(0).unsqueeze(0)

    # 广播归一化
    data_norm = (data - data_min) / diff.unsqueeze(0).unsqueeze(0)
    
    return data_norm

def collate_fn(batch):
    batch.sort(key=lambda item: len(item[0]), reverse=True)
    bvp_batch,batch_ges=zip(*batch)
    lengths=[len(x) for x in bvp_batch]
    batch_bvp=pad_sequence(bvp_batch,batch_first=True)
    batch_ges=torch.tensor(batch_ges)
    bvp_len=torch.tensor(lengths)
    return batch_bvp,batch_ges,bvp_len

def get_label_mapping(setting):

    if setting == 1:
        # Setting 1
        label_map = {
            'Push&Pull': 0,
            'Sweep': 1,
            'Clap': 2,
            'Slide': 3,

            # 合并 N
            'Draw-N(H)': 4,
            'Draw-N(V)': 4,

            # 合并 Z (Zigzag)
            'Draw-Zigzag(H)': 5,
            'Draw-Zigzag(V)': 5,
        }

    elif setting == 2:
        # Setting 2
        label_map = {
            'Push&Pull': 0,
            'Sweep': 1,
            'Clap': 2,
            'Slide': 3,

            # 合并 O
            'Draw-O(H)': 4,
            'Draw-O(V)': 4,

            # 合并 Z
            'Draw-Zigzag(H)': 5,
            'Draw-Zigzag(V)': 5,
        }

    else:
        raise ValueError("SETTING must be 1 or 2")

    return label_map

def build_dataloader(batch_size=32, num_workers=0,setting=1,mode='in_domain',target_domain=None):
    """
    Args:
        mode: 'in_domain', 'cross_user', 'cross_location', 'cross_room'
        target_domain: 当 mode 为跨域模式时，指定作为测试集的域值
    """

    if mode=='in_domain' and setting!=1:
        raise ValueError("In-domain setting must be 1")
        
    dataset = WidarGestureDataset(root_dir,setting)

    train_indices=[]
    test_indices=[]

    if mode=='in_domain':

        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    else:
        print(f"构建跨域数据集，模式: {mode}, 目标域: {target_domain}")

        for i in range(len(dataset)):
            domain_info=dataset.get_domain_label(i)
            if mode=='cross_user':
                val=domain_info['user']
            elif mode=='cross_location':
                val=domain_info['location']
            elif mode=='cross_room':
                val=domain_info['room']
            else:
                raise ValueError("mode must be 'in_domain', 'cross_user', 'cross_location' or 'cross_room'")
            
            if val==target_domain:
                test_indices.append(i)
            else:
                train_indices.append(i)

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    print(f"数据加载完成，训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    return train_loader, test_loader
if __name__ == "__main__":
    build_dataloader(batch_size=32, num_workers=0,setting=1,mode='in_domain',target_domain=None)