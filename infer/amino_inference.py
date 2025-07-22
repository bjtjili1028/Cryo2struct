"""
author: nabin 
timestamp: Mon Sep 04 2023 06:28 PM

AMINO PREDICTION
"""
import json
import math
from copy import deepcopy

import mrcfile
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from einops import rearrange

# 引入 Transformer UNet 模組
from self_attention_cv.UnetTr.modules import TranspConv3DBlock, BlueBlock, Conv3DBlock
from self_attention_cv.UnetTr.volume_embedding import Embeddings3D
from self_attention_cv.transformer_vanilla import TransformerBlock

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from argparse import ArgumentParser

import sys

import warnings
warnings.filterwarnings("ignore")

# 定義數據處理的參數

box_size = 32  # Expected Dimensions to pass to Transformer Unet
core_size = 20  # core of the image where we dnt have to worry about boundary issues

BATCH_SIZE = 1  # for now # 當前批次大小
DATALOADERS = 1 # 數據加載器的數量

data_splits = list() # 存儲數據集切分的列表
collect_pred_probs = dict() # 用於存儲預測的概率
idx_vals = list() # 存儲索引值
raw_logits = list() # 存儲原始的預測logits
idx_val_list = list() # 存儲最終的索引值列表


def prepare_data(dataset_dir, density_map_name):
    # 準備數據：從資料夾中獲取數據切分列表
    data_splits_old = [splits for splits in os.listdir(dataset_dir)]
    for arr in range(len(data_splits_old)):
        # 根據密度圖名稱和切分索引來生成文件名稱
        data_splits.append(f"{density_map_name}_{arr}.npz")



class CryoData(Dataset):
    # 定義PyTorch的數據集類別
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # 返回數據集的大小
        return len(data_splits)

    def __getitem__(self, idx):
        # 獲取對應的數據文件名
        cryodata = data_splits[idx]
        cryodata = cryodata.strip("\n")
        
        # 加載數據
        loaded_data = np.load(f"{self.root}/{cryodata}")
        # 提取蛋白質網格數據
        protein_manifest = loaded_data['protein_grid']
        # 轉換為Tensor
        protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
        return [protein_torch] # 返回處理後的蛋白質數據


# 定義 Transformer 編碼器
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout, extract_layers, dim_linear_block):
        super().__init__()
        self.layer = nn.ModuleList()
        self.extract_layers = extract_layers

        # make TransformerBlock device
        self.block_list = nn.ModuleList()
        for _ in range(num_layers):
            # 添加 TransformerBlock 層
            self.block_list.append(
                TransformerBlock(dim=embed_dim, heads=num_heads, dim_linear_block=dim_linear_block, dropout=dropout,
                                 prenorm=True))

    def forward(self, x):
        extract_layers = []
        for depth, layer_block in enumerate(self.block_list):
            x = layer_block(x) # 前向傳播
            if (depth + 1) in self.extract_layers:
                extract_layers.append(x) # 根據需要提取特定層的輸出
        return extract_layers


class Transformer_UNET(nn.Module):
    # 定義 Transformer U-Net 模型
    def __init__(self, img_shape=(64, 64, 64), input_dim=1, output_dim=21, embed_dim=768, patch_size=16,
                 num_heads=12, dropout=0.0, ext_layers=[3, 6, 9, 12], norm="instance", base_filters=16,
                 dim_linear_block=3072):
        super().__init__()
        self.num_layers = 12
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.ext_layers = ext_layers
        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # 根據選擇進行歸一化
        self.norm = nn.BatchNorm3d if norm == 'batch' else nn.InstanceNorm3d

        # 3D嵌入層
        self.embed = Embeddings3D(input_dim=input_dim, embed_dim=embed_dim, cube_size=img_shape,
                                  patch_size=patch_size, dropout=dropout)

        # Transformer編碼器
        self.transformer = TransformerEncoder(embed_dim, num_heads, self.num_layers, dropout, ext_layers,
                                              dim_linear_block=dim_linear_block)
       
        # 卷積層
        self.init_conv = Conv3DBlock(input_dim, base_filters, double=True, norm=self.norm)

        # blue block

        self.z3_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 2, layers=3)
        self.z6_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 4, layers=2)
        self.z9_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 8, layers=1)

        # Green block

        self.z12_deconv = TranspConv3DBlock(embed_dim, base_filters * 8)
        self.z9_deconv = TranspConv3DBlock(base_filters * 8, base_filters * 4)
        self.z6_deconv = TranspConv3DBlock(base_filters * 4, base_filters * 2)
        self.z3_deconv = TranspConv3DBlock(base_filters * 2, base_filters)

        # yellow block

        self.z9_conv = Conv3DBlock(base_filters * 8 * 2, base_filters * 8, double=True, norm=self.norm)
        self.z6_conv = Conv3DBlock(base_filters * 4 * 2, base_filters * 4, double=True, norm=self.norm)
        self.z3_conv = Conv3DBlock(base_filters * 2 * 2, base_filters * 2, double=True, norm=self.norm)

        # out convolutions

        self.out_conv = nn.Sequential(
            # last yellow conv block
            Conv3DBlock(base_filters * 2, base_filters, double=True, norm=self.norm),

            # grey block, final classification
            nn.Conv3d(base_filters, output_dim, kernel_size=1, stride=1))
        

    def forward(self, x):
        transformer_input = self.embed(x)
        # 將8*768 轉回 16×16×16×768
        z3, z6, z9, z12 = map(
            lambda t: rearrange(t, 'b (x y z) d -> b d x y z', x=self.patch_dim[0], y=self.patch_dim[1],
                                z=self.patch_dim[2]), self.transformer(transformer_input))

        # Blue convs
        z0 = self.init_conv(x)
        z3 = self.z3_blue_conv(z3)
        z6 = self.z6_blue_conv(z6)
        z9 = self.z9_blue_conv(z9)

        # Green blocks for z12
        # 將 Z_12 (16×16×16×768) 還原成 32×32×32×768
        z12 = self.z12_deconv(z12)

        # concat + yellow conv

        y = torch.cat([z12, z9], dim=1)
        y = self.z9_conv(y) # Conv3D 3×3×3, 輸出 (32×32×32×16)

        # Green blocks for z6
        y = self.z9_deconv(y)

        # concat + yellow conv
        y = torch.cat([y, z6], dim=1)
        y = self.z6_conv(y)

        # Green block for z3
        y = self.z6_deconv(y)

        # concat + yellow conv

        y = torch.cat([y, z3], dim=1)

        y = self.z3_conv(y)

        y = self.z3_deconv(y)
        y = torch.cat([y, z0], dim=1)
        return self.out_conv(y)


class VoxelClassify(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer_UNET(**model_kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data):
        x = self.model(data)
        return x
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        protein_data = batch[0]
        protein_data = torch.unsqueeze(protein_data, 1) # 增加額外的維度，符合模型輸入要求
        pred = self(protein_data)
        
        s = torch.softmax(pred[0], dim=0) # 計算softmax概率
        s_permute = torch.permute(s, (1, 2, 3, 0)) # 重排概率值
        idx_val_np = np.empty(shape=(32, 32, 32), dtype='S30') # 用來存儲最終的預測值

        # used softmax for hidden markov model.
        # 使用softmax的結果來做最終預測
        a = torch.argmax(pred[0], dim=0)# 取最大概率的索引

        # 儲存每一個預測結果
        for i in range(len(s_permute)):
            for j in range(len(s_permute[i])):
                for k in range(len(s_permute[i][j])):
                    val_prob = s_permute[i][j][k]
                    collect_pred_probs[f'{batch_idx}_{i}_{j}_{k}'] = val_prob
                    v = f'{batch_idx}_{i}_{j}_{k}'
                    idx_val_np[i][j][k] = v
        idx_val_list.append(idx_val_np)
        return a # 返回最大概率的預測值
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        return parser


def infer_classifier(density_map_splits_dir, input_data_dir, density_map_name, amino_checkpoint, infer_run_on, infer_on_gpu):
    pl.seed_everything(42) # 設置隨機種子，確保實驗可重現
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VoxelClassify.add_model_specific_args(parser)

    prepare_data(dataset_dir=density_map_splits_dir, density_map_name=density_map_name) # 準備數據
    dataset = CryoData(density_map_splits_dir) # 加載數據
    test_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False,
                             num_workers=1) # 創建數據加載器

    args, unknown = parser.parse_known_args()
    args.detect_anomaly=True # 啟用異常檢測
    args.enable_model_summary = True # 啟用模型摘要
    if infer_run_on == "gpu":
        args.accelerator = "gpu"
        args.devices = [infer_on_gpu] # 設置使用的GPU設備
    else:
        args.accelerator = "cpu" # 使用CPU運行

    # 創建模型
    model = VoxelClassify(learning_rate=1e-4, img_shape=(32, 32, 32), input_dim=1, output_dim=21, embed_dim=768,
                          patch_size=16, num_heads=12, dropout=0.0, ext_layers=[3, 6, 9, 12], norm="instance", 
                          base_filters=16, dim_linear_block=3072)

    # 設置訓練器
    trainer = pl.Trainer.from_argparse_args(args)
    
    # 預測結果
    predicts = trainer.predict(model, dataloaders=test_loader, ckpt_path=amino_checkpoint)
    
    # 轉換為numpy數組
    for pred in range(len(predicts)):
        predicts[pred] = predicts[pred].numpy() # 轉換為numpy數組
    
    # 保存 predicts 至 .npy 檔案
    # npy_filename = f"{input_data_dir}/{density_map_name}/{density_map_name}_amino_predicts.npy"
    # np.save(npy_filename, np.array(predicts, dtype=object))  # 使用 dtype=object 以保留變長數據
    # print(f"Predictions saved as:\n - {npy_filename}")
    
    # 讀取mrc檔案
    org_map = f"{input_data_dir}/{density_map_name}/emd_normalized_map.mrc" # 原始密度圖
    org_map = mrcfile.open(org_map, mode='r') # 讀取mrc檔案

    recon, idx_val_mat = reconstruct_map(manifest=predicts, idx_val_np=idx_val_list, image_shape=org_map.data.shape) # 重建圖像
    filename = "amino_predicted.mrc" # 預測結果保存的檔案名
    outfilename = f"{input_data_dir}/{density_map_name}/{density_map_name}_{filename}"
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(recon) # 保存預測結果到mrc檔案
        mrc.voxel_size = 1
        mrc.header.origin = org_map.header.origin
        mrc.close()

    # save the probabilities
    # 保存預測概率
    file_prob = f"{input_data_dir}/{density_map_name}/{density_map_name}_probabilities_amino.txt"
    save_probs(outfilename, idx_val_mat, file_prob) # 保存概率值


def reconstruct_map(manifest, idx_val_np, image_shape):
    # 根據Transformer Unet的輸出重建完整的蛋白質圖像
    # takes the output of Transformer Unet and reconstructs the full dimension of the protein
    extract_start = int((box_size - core_size) / 2)
    extract_end = int((box_size - core_size) / 2) + core_size
    dimentions = get_manifest_dimensions(image_shape)

    reconstruct_image = np.zeros((dimentions[0], dimentions[1], dimentions[2]))

    idx_val_mat = np.empty(shape=(dimentions[0], dimentions[1], dimentions[2]), dtype='S30')

    counter = 0
    for z_steps in range(int(dimentions[2] / core_size)):
        for y_steps in range(int(dimentions[1] / core_size)):
            for x_steps in range(int(dimentions[0] / core_size)):
                reconstruct_image[x_steps * core_size:(x_steps + 1) * core_size,
                y_steps * core_size:(y_steps + 1) * core_size, z_steps * core_size:(z_steps + 1) * core_size] = \
                    manifest[counter][extract_start:extract_end, extract_start:extract_end,
                    extract_start:extract_end]

                idx_val_mat[x_steps * core_size:(x_steps + 1) * core_size,
                y_steps * core_size:(y_steps + 1) * core_size, z_steps * core_size:(z_steps + 1) * core_size] = \
                    idx_val_np[counter][extract_start:extract_end, extract_start:extract_end,
                    extract_start:extract_end]

                counter += 1
    float_reconstruct_image = np.array(reconstruct_image, dtype=np.float32)
    float_reconstruct_image = float_reconstruct_image[:image_shape[0], :image_shape[1], :image_shape[2]]
    idx_val_np_mat = idx_val_mat[:image_shape[0], :image_shape[1], :image_shape[2]]
    return float_reconstruct_image, idx_val_np_mat

# 計算manifest的維度，這樣可以確保重建時不會超出邊界
def get_manifest_dimensions(image_shape):
    dimensions = [0, 0, 0]
    dimensions[0] = math.ceil(image_shape[0] / core_size) * core_size
    dimensions[1] = math.ceil(image_shape[1] / core_size) * core_size
    dimensions[2] = math.ceil(image_shape[2] / core_size) * core_size
    return dimensions


def get_xyz(idx, voxel, origin):
    return (idx * voxel) + origin

# 保存預測概率到文件中
def save_probs(mrc_file, idx_file, file_prob):
    mrc_map = mrcfile.open(mrc_file, mode='r')
    x_origin = mrc_map.header.origin['x']
    y_origin = mrc_map.header.origin['y']
    z_origin = mrc_map.header.origin['z']
    x_voxel = mrc_map.voxel_size['x']
    y_voxel = mrc_map.voxel_size['y']
    z_voxel = mrc_map.voxel_size['z']
    mrc_data = deepcopy(mrc_map.data)
    with open(file_prob, "w") as f:
        for k in range(len(mrc_data[2])):
            for j in range(len(mrc_data[1])):
                for i in range(len(mrc_data[0])):
                    try:
                        if mrc_data[i][j][k] > 0:
                            ids = idx_file[i][j][k]
                            x = round(get_xyz(k, x_voxel, x_origin), 3)
                            y = round(get_xyz(j, y_voxel, y_origin), 3)
                            z = round(get_xyz(i, z_voxel, z_origin), 3)
                            ids = ids.decode()
                            value = collect_pred_probs[ids]
                            lst = value.tolist()
                            lst.insert(0,[x,y,z])
                            json_dump = json.dumps(lst)
                            final = json_dump[1:-1]
                            f.writelines(final)
                            f.writelines('\n')
                    except UnicodeDecodeError:
                        print("Error", i, j, k)
                        pass
                    except IndexError:
                        pass


if __name__ == "__main__":

    density_map_splits_dir = sys.argv[1]
    input_data_dir = sys.argv[2]
    density_map = sys.argv[3]
    amino_checkpoint = sys.argv[4]
    infer_run_on = sys.argv[5]
    infer_run_gpu = int(sys.argv[6])
    infer_classifier(density_map_splits_dir=density_map_splits_dir, input_data_dir=input_data_dir, density_map_name=density_map, 
                          amino_checkpoint=amino_checkpoint, infer_run_on=infer_run_on, infer_on_gpu=infer_run_gpu)