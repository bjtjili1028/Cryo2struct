"""
author: nabin 
timestamp: Tue Jan 02 2024 04:00 PM
"""

import os
import numpy as np

import torch # 引入PyTorch深度學習框架
import torch.nn as nn # 用於神經網絡層
from einops import rearrange  # 用於重排列張量的庫


# 引入自定義模組中的層和組件，建立在使用PyTorch的基礎上 (需要安裝 self-attention-cv)
# https://github.com/The-AI-Summer/self-attention-cv/tree/main
from self_attention_cv.UnetTr.modules import TranspConv3DBlock, BlueBlock, Conv3DBlock
from self_attention_cv.UnetTr.volume_embedding import Embeddings3D
from self_attention_cv.transformer_vanilla import TransformerBlock

import pytorch_lightning as pl # 用於PyTorch的高層API
from pytorch_lightning.loggers import WandbLogger # 用於Wandb記錄訓練過程
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor # 用於早期停止、模型檢查點和學習率監控
from torch.utils.data import DataLoader # 用於加載數據
from torch.utils.data import Dataset # 用於創建自定義數據集
from argparse import ArgumentParser # 用於解析命令行參數
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, FBetaScore  # 用於多個指標計算

AVAIL_GPUS = 6
NUM_NODES = 1 # 使用的節點數量
BATCH_SIZE = 2 * 6 * 1 # batch size * available GPU * number of nodes
DATALOADERS = 6 # 數據加載器數量
STRATEGY = "ddp_find_unused_parameters_false" # 分布式訓練策略
ACCELERATOR = "gpu"
GPU_PLUGIN = "ddp_sharded"  # 使用DDP分片插件
EPOCHS = 100 # 訓練輪數
CHECKPOINT_PATH = "amino_checkpoint" # 檢查點保存路徑
os.makedirs(CHECKPOINT_PATH, exist_ok=True)  # 如果路徑不存在則創建

DATASET_DIR = "/input/" # 資料集的目錄
TRAIN_SUB_GRIDS = "train_sub_grids"  # 訓練數據的子目錄
VALID_SUB_GRIDS = "valid_sub_grids"  # 驗證數據的子目錄

# 讀取訓練和驗證數據集列表
file = open(os.path.join(DATASET_DIR, 'train_splits.txt'))
train = file.readlines()
print("Training Data file found and the number of protein graph splits are:", len(train))

file = open(os.path.join(DATASET_DIR, 'valid_splits.txt'))
valid = file.readlines()
print("Valid Data file found and the number of protein graph splits are:", len(valid))

# 計算模型中需要訓練的參數數量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 定義訓練數據集
class CryoData(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.trarget_transform = target_transform

    def __len__(self):
        return len(train) # 訓練集的長度

    def __getitem__(self, idx):
        cryodata = train[idx]
        cryodata = cryodata.strip("\n") # 讀取每個數據文件
        loaded_data = np.load(f"{DATASET_DIR}/{TRAIN_SUB_GRIDS}/{cryodata}") # 加載網格數據

        protein_manifest = loaded_data['protein_grid'] # 蛋白質網格
        protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
        amino_manifest = loaded_data['amino_grid'] # 氨基酸網格
        amino_torch = torch.from_numpy(amino_manifest).type(torch.FloatTensor)
        return [protein_torch, amino_torch]  # 返回處理後的蛋白質和氨基酸網格
 
# 定義驗證數據集
class CryoData_valid(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.trarget_transform = target_transform

    def __len__(self):
        return len(valid)

    def __getitem__(self, idx):
        cryodata = valid[idx]
        cryodata = cryodata.strip("\n")
        loaded_data = np.load(f"{DATASET_DIR}/{VALID_SUB_GRIDS}/{cryodata}") # 加載網格數據
        protein_manifest = loaded_data['protein_grid'] # 蛋白質網格
        protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
        amino_manifest = loaded_data['amino_grid'] # 氨基酸網格
        amino_torch = torch.from_numpy(amino_manifest).type(torch.FloatTensor)
        print(f"Protein Grid Shape: {protein_manifest.shape}, Amino Grid Shape: {amino_manifest.shape}")
        return [protein_torch, amino_torch] # 返回處理後的蛋白質和氨基酸網格

# 定義Transformer編碼器
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout, extract_layers, dim_linear_block):
        super().__init__()
        self.layer = nn.ModuleList()
        self.extract_layers = extract_layers

        self.block_list = nn.ModuleList()
        for _ in range(num_layers):
            self.block_list.append(
                TransformerBlock(dim=embed_dim, heads=num_heads, dim_linear_block=dim_linear_block, dropout=dropout,
                                 prenorm=True)) # 定義Transformer層
 
    def forward(self, x):
        extract_layers = []
        for depth, layer_block in enumerate(self.block_list):
            x = layer_block(x)
            if (depth + 1) in self.extract_layers:
                extract_layers.append(x) # 根據需要提取某些層
        return extract_layers

# 定義Transformer UNET模型
class Transformer_UNET(nn.Module):
    def __init__(self, img_shape=(32, 32, 32), input_dim=1, output_dim=21, embed_dim=768, patch_size=16,
                 num_heads=6, dropout=0.1, ext_layers=[3, 6, 9, 12], norm="instance", base_filters=16,
                 dim_linear_block=3072):
        super().__init__()
        self.num_layers = 12 # Transformer層數
        self.input_dim = input_dim # 輸入單一維度
        self.output_dim = output_dim # 輸出為21維向量，表示氨基酸的21個類別
        self.embed_dim = embed_dim # 768
        self.img_shape = img_shape  # 輸入資料維度
        self.patch_size = patch_size # 補丁數量
        self.num_heads = num_heads # 注意力機制的頭數
        self.dropout = dropout  # 丟棄率為0.1
        self.ext_layers = ext_layers # 需要提取的層
        self.patch_dim = [int(x / patch_size) for x in
                          img_shape]

        # 選擇歸一化方法
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

        # out convolutions (最終卷積層)
        self.out_conv = nn.Sequential(
            # last yellow conv block
            Conv3DBlock(base_filters * 2, base_filters, double=True, norm=self.norm),

            # brown block, final classification(分類)
            nn.Conv3d(base_filters, output_dim, kernel_size=1, stride=1))

    def forward(self, x):
        transformer_input = self.embed(x)
        z3, z6, z9, z12 = map(
            lambda t: rearrange(t, 'b (x y z) d -> b d x y z', x=self.patch_dim[0], y=self.patch_dim[1],
                                z=self.patch_dim[2]), self.transformer(transformer_input))

        # Blue convs
        z0 = self.init_conv(x)
        z3 = self.z3_blue_conv(z3)
        z6 = self.z6_blue_conv(z6)
        z9 = self.z9_blue_conv(z9)

        # Green blocks for z12
        z12 = self.z12_deconv(z12)

        # concat + yellow conv
        y = torch.cat([z12, z9], dim=1)
        y = self.z9_conv(y)

        # Green blocks for z6
        y = self.z9_deconv(y)

        # concat + yellow conv (連接並進行黃色卷積)
        y = torch.cat([y, z6], dim=1)
        # y = torch.cat([attention_values, z6], dim=1)
        y = self.z6_conv(y)

        # Green block for z3
        y = self.z6_deconv(y)

        # concat + yellow conv
        y = torch.cat([y, z3], dim=1)
        y = self.z3_conv(y)
        y = self.z3_deconv(y)
        y = torch.cat([y, z0], dim=1)
        return self.out_conv(y) # 返回最終的預測結果

# 計算每個類別的權重，用於處理類別不平衡問題
def calc_ce_weights(batch):
    # 計算每一類別的樣本數
    y_zeros = (batch == 0.).sum()
    y_ones = (batch == 1.).sum()
    y_two = (batch == 2.).sum()
    y_three = (batch == 3.).sum()
    y_four = (batch == 4.).sum()
    y_five = (batch == 5.).sum()
    y_six = (batch == 6.).sum()
    y_seven = (batch == 7.).sum()
    y_eight = (batch == 8.).sum()
    y_nine = (batch == 9.).sum()
    y_ten = (batch == 10.).sum()
    y_eleven = (batch == 11.).sum()
    y_twelve = (batch == 12.).sum()
    y_thirteen = (batch == 13.).sum()
    y_fourteen = (batch == 14.).sum()
    y_fifteen = (batch == 15.).sum()
    y_sixteen = (batch == 16.).sum()
    y_seventeen = (batch == 17.).sum()
    y_eighteen = (batch == 18.).sum()
    y_nineteen = (batch == 19.).sum()
    y_twenty = (batch == 20.).sum()
    nSamples = [y_zeros, y_ones, y_two, y_three, y_four, y_five, y_six, y_seven, y_eight, y_nine, y_ten, y_eleven,
                y_twelve,
                y_thirteen, y_fourteen, y_fifteen, y_sixteen, y_seventeen, y_eighteen, y_nineteen, y_twenty]
    
    # 計算每個類別的權重，較少的類別將會分配較高的權重
    normedWeights_1 = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = [x + 1e-5 for x in normedWeights_1] # 避免除零錯誤
    balance_weights = torch.FloatTensor(normedWeights).to("cuda")
    return balance_weights


class VoxelClassify(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, **model_kwargs):
        super().__init__()
        # 保存超參數，讓模型可以在訓練後進行回溯
        self.save_hyperparameters()
        # 初始化Transformer_UNET模型
        self.model = Transformer_UNET(**model_kwargs)
        # 定義交叉熵損失函數，將會用來計算訓練過程中的損失
        self.loss_fn = nn.CrossEntropyLoss()

        # 定義宏平均指標集合
        self.metrics_macro = MetricCollection([Accuracy(task='multiclass', num_classes=21, average='macro', mdmc_average="global"),
                                               Precision(task='multiclass', num_classes=21, average='macro', mdmc_average="global"),
                                               Recall(task='multiclass', num_classes=21, average='macro', mdmc_average="global"),
                                               F1Score(task='multiclass', num_classes=21, average='macro', mdmc_average="global"),
                                               FBetaScore(task='multiclass', num_classes=21, average='macro', mdmc_average="global")])
        
        # 定義加權平均指標集合
        self.metrics_weighted = MetricCollection([Accuracy(task='multiclass', num_classes=21, average='weighted', mdmc_average="global"),
                                                  Precision(task='multiclass', num_classes=21, average='weighted', mdmc_average="global"),
                                                  Recall(task='multiclass', num_classes=21, average='weighted', mdmc_average="global"),
                                                  F1Score(task='multiclass', num_classes=21, average='weighted', mdmc_average="global"),
                                                  FBetaScore(task='multiclass', num_classes=21, average='weighted', mdmc_average="global")])
        """
        self.metrics_micro = MetricCollection([Accuracy(num_classes=21, average='micro', mdmc_average="global"),
                                               Precision(num_classes=21, average='micro', mdmc_average="global"),
                                               Recall(num_classes=21, average='micro', mdmc_average="global"),
                                               F1Score(num_classes=21, average='micro', mdmc_average="global"),
                                               FBetaScore(num_classes=21, average='micro', mdmc_average="global")])
        """
        
        # 克隆宏平均指標並用於訓練、驗證和測試
        self.train_metrics_macro = self.metrics_macro.clone(prefix="train_macro_")
        self.valid_metrics_macro = self.metrics_macro.clone(prefix="valid_macro_")
        self.test_metrics_macro = self.metrics_macro.clone(prefix="test_macro_")

        self.train_metrics_weighted = self.metrics_weighted.clone(prefix="train_weighted_")
        self.valid_metrics_weighted = self.metrics_weighted.clone(prefix="valid_weighted_")
        self.test_metrics_weighted = self.metrics_weighted.clone(prefix="test_weighted_")
        """
        self.train_metrics_micro = self.metrics_micro.clone(prefix="train_micro_")
        self.valid_metrics_micro = self.metrics_micro.clone(prefix="valid_micro_")
        self.test_metrics_micro = self.metrics_micro.clone(prefix="test_micro_")
        
        """
    # 前向傳遞函數，將數據傳遞給模型並返回預測結果
    def forward(self, data):
        # 將數據傳遞給Transformer_UNET模型
        x = self.model(data) 
        return x

    def configure_optimizers(self):
        # 使用NAdam優化器
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.hparams.learning_rate)
        
        # 使用ReduceLROnPlateau學習率調度器來根據驗證損失減少學習率
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            mode='min', 
            factor=0.1,
            patience=10, 
            eps=1e-10, 
            verbose=True)
        
        # 監控的指標是訓練損失
        metric_to_track = 'train_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': metric_to_track
        }

    def training_step(self, batch, batch_idx):
        protein_data, atom_data = batch[0], batch[1] # 提取蛋白質數據和原子標籤
        protein_data = torch.unsqueeze(protein_data, 1) # 增加一個維度以符合模型要求
        y_hat = self.forward(protein_data) # 獲得模型預測結果
        
        # 計算加權的交叉熵損失
        balance_weights = calc_ce_weights(atom_data) # 計算每個類別的權重
        loss_fn_train = nn.CrossEntropyLoss(weight=balance_weights) # 使用加權的交叉熵損失函數
        loss = loss_fn_train(y_hat, atom_data.long()) # 計算損失
        
        # 計算並記錄宏平均指標
        metric_log_macro = self.train_metrics_macro(y_hat, atom_data.int())
        
        # metric_log_micro = self.train_metrics_micro(y_hat, amino_data.int())
        
        # 計算並記錄加權平均指標
        metric_log_weighted = self.train_metrics_weighted(y_hat, atom_data.int())
        
        # 記錄訓練損失和指標
        self.log_dict(metric_log_macro,on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(metric_log_micro)
        self.log_dict(metric_log_weighted, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss  # 返回計算出的損失

    def validation_step(self, batch, batch_idx):
        protein_data, atom_data = batch[0], batch[1] # 提取蛋白質數據和原子標籤
        protein_data = torch.unsqueeze(protein_data, 1) # 增加一個維度以符合模型要求
        y_hat = self.forward(protein_data) # 獲得模型預測結果
        loss = self.loss_fn(y_hat, atom_data.long()) # 計算損失
        # loss = loss_fn_train(y_hat, batch.y.long())
        
        # 計算並記錄宏平均指標
        metric_log_macro = self.valid_metrics_macro(y_hat, atom_data.int())
        
        # metric_log_micro = self.valid_metrics_micro(y_hat, amino_data.int())
        
        # 計算並記錄加權平均指標
        metric_log_weighted = self.valid_metrics_weighted(y_hat, atom_data.int())
        
        # 記錄驗證損失和指標
        self.log_dict(metric_log_macro, on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(metric_log_micro)
        self.log_dict(metric_log_weighted, on_step=True, on_epoch=True, sync_dist=True)
        self.log('valid_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        protein_data, atom_data = batch[0], batch[1] # 提取蛋白質數據和原子標籤
        protein_data = torch.unsqueeze(protein_data, 1) # 增加一個維度以符合模型要求
        y_hat = self.forward(protein_data) # 獲得模型預測結果
        loss = self.loss_fn(y_hat, atom_data.long()) # 計算損失
        # loss = loss_fn_train(y_hat, batch.y.long())
        
        # 計算並記錄宏平均指標
        metric_log_macro = self.test_metrics_macro(y_hat, atom_data.int())
        
        # metric_log_micro = self.test_metrics_micro(y_hat, amino_data.int())
        
        # 計算並記錄加權平均指標
        metric_log_weighted = self.test_metrics_weighted(y_hat, atom_data.int())
        
        # 記錄測試損失和指標
        self.log_dict(metric_log_macro,  on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(metric_log_micro)
        self.log_dict(metric_log_weighted, on_step=True, on_epoch=True, sync_dist=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    # 定義模型超參數 (預設學習率為1e-4)
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        return parser

# 訓練過程的主函數
def train_node_classifier():
    pl.seed_everything(42) # 設置隨機種子，保證實驗可重複
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VoxelClassify.add_model_specific_args(parser)
    # training specific args
    # 設定多GPU訓練參數
    parser.add_argument('--multi_gpu_backend', type=str, default=STRATEGY,
                        help="Backend to use for multi-GPU training")
    parser.add_argument('--advance_gpu_plugins', type=str, default=GPU_PLUGIN,
                        help="Shard the optimizer and model into multiple gpus")
    parser.add_argument('--modify_precision', type=int, default=16, help="Precision to improve training")
    parser.add_argument('--num_gpus', type=int, default=AVAIL_GPUS,
                        help="Number of GPUs to use (e.g. -1 = all available GPUs)")
    parser.add_argument('--nodes', type=int, default=NUM_NODES, help="Number of nodes to use")
    parser.add_argument('--num_epochs', type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int,
                        help="effective_batch_size = batch_size * num_gpus * num_nodes")
    parser.add_argument('--num_dataloader_workers', type=int, default=DATALOADERS)
    parser.add_argument('--entity_name', type=str, default='nabingiri', help="Weights and Biases entity name")
    parser.add_argument('--project_name', type=str, default='github_test_train',
                        help="Weights and Biases project name")
    parser.add_argument('--save_dir', type=str, default=CHECKPOINT_PATH, help="Directory in which to save models")
    parser.add_argument('--unit_test', type=int, default=False,
                        help="helps in debug, this touches all the parts of code."
                             "Enter True or num of batch you want to send, " "eg. 1 or 7")
    args = parser.parse_args()

    # 設置訓練參數
    args.strategy = args.multi_gpu_backend
    args.devices = args.num_gpus
    args.num_nodes = args.nodes
    args.accelerator = "gpu"
    args.max_epochs = args.num_epochs
    args.precision = args.modify_precision
    args.fast_dev_run = args.unit_test
    args.log_every_n_steps = 10
    # args.detect_anomaly = True
    args.terminate_on_nan = True
    args.enable_model_summary = True
    args.weights_summary = "full"

    # to resume training from saved checkpoint:
    # args.resume_from_checkpoint = "CHECKPOINT_PATH/amino-epoch=02-valid_loss=0.192797.ckpt" 

    # 創建數據集和數據加載器
    dataset = CryoData(DATASET_DIR)
    dataset_valid = CryoData_valid(DATASET_DIR)

    train_data = dataset
    val_data = dataset_valid
    test_data = dataset_valid

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False,
                              num_workers=args.num_dataloader_workers)
    valid_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False,
                              num_workers=args.num_dataloader_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False,
                             num_workers=args.num_dataloader_workers)

    model = VoxelClassify(learning_rate=1e-4, img_shape=(32, 32, 32), input_dim=1, output_dim=21, embed_dim=768,
                          patch_size=16,
                          num_heads=6, dropout=0.1, ext_layers=[3, 6, 9, 12], norm="instance", base_filters=16,
                          dim_linear_block=3072)
    # 計算模型的參數數量並顯示
    print("Model's trainable parameters", count_parameters(model))

    # 使用PyTorch Lightning的trainer來訓練模型
    trainer = pl.Trainer.from_argparse_args(args)

    # early_stopping_callback = EarlyStopping(monitor='valid_loss', mode='min', min_delta=0.0, patience=5)
    # 設定模型檢查點回調函數
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_top_k=5, dirpath=args.save_dir,
                                          filename='amino-{epoch:02d}-{valid_loss:.6f}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # trainer.callbacks = [checkpoint_callback, lr_monitor, early_stopping_callback]
    trainer.callbacks = [checkpoint_callback, lr_monitor]
    
    # 使用Weights and Biases來記錄訓練過程
    logger = WandbLogger(project=args.project_name, entity=args.entity_name, offline=False)
    trainer.logger = logger
    
    # 訓練模型
    trainer.fit(model, train_loader, valid_loader)
    
    # 在測試集上進行測試
    trainer.test(dataloaders=test_loader, ckpt_path='best')

# 啟動訓練過程
if __name__ == "__main__":
    train_node_classifier()
