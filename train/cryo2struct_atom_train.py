"""
author: nabin 
timestamp: Tue Jan 02 2024 02:00 PM
"""

import os
import numpy as np

import torch # 用於深度學習框架
import torch.nn as nn # 用於神經網絡層
from einops import rearrange # 用於數據重排（例如，重排列張量）

# 引入注意力機制中的層和組件，建立在使用pytorch的基礎上 (pip install self-attention-cv==1.2.3)
# https://github.com/The-AI-Summer/self-attention-cv/tree/main
from self_attention_cv.UnetTr.modules import TranspConv3DBlock, BlueBlock, Conv3DBlock 
from self_attention_cv.UnetTr.volume_embedding import Embeddings3D
from self_attention_cv.transformer_vanilla import TransformerBlock

import pytorch_lightning as pl # 用於PyTorch的高階API
from pytorch_lightning.loggers import WandbLogger # 用於Wandb記錄訓練過程
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor # 用於早期停止、模型檢查點和學習率監控
from torch.utils.data import DataLoader # 用於加載數據
from torch.utils.data import Dataset # 用於創建自定義數據集
from argparse import ArgumentParser # 用於解析命令行參數
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, FBetaScore # 用於多個指標計算

# 設置訓練參數
AVAIL_GPUS = 6 # 可用的gpu數量
NUM_NODES = 2 # 節點數量
BATCH_SIZE = 4 * 6 * 2 # batch size * available GPU * number of nodes
DATALOADERS = 6 # 數據加載器數量
STRATEGY = "ddp_find_unused_parameters_false" # 分布式訓練策略
ACCELERATOR = "gpu" # 使用gpu加速
GPU_PLUGIN = "ddp_sharded" # ddp 分片插件
EPOCHS = 100 # 訓練輪數
CHECKPOINT_PATH = "atom_checkpoint" # 檢查點保存路徑，從參數檔抓取
os.makedirs(CHECKPOINT_PATH, exist_ok=True) # 如果路徑不存在則創立

DATASET_DIR = "/input/" # 資料來源資料夾
TRAIN_SUB_GRIDS = "train_sub_grids" # 訓練目錄
VALID_SUB_GRIDS = "valid_sub_grids" # 驗證目錄

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
        return len(train)  # 訓練集的長度

    def __getitem__(self, idx):
        cryodata = train[idx]
        cryodata = cryodata.strip("\n") # 讀取每個數據文件
        loaded_data = np.load(f"{DATASET_DIR}/{TRAIN_SUB_GRIDS}/{cryodata}") # 加載網格數據

        protein_manifest = loaded_data['protein_grid'] # 蛋白質網格
        protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
        
        atom_manifest = loaded_data['atom_grid'] # 原子網格
        atom_torch = torch.from_numpy(atom_manifest).type(torch.FloatTensor)
        # esm_embeds = loaded_data['embeds']
        # esm_embeds_torch = torch.from_numpy(esm_embeds).type(torch.FloatTensor)
        print(f"Protein Grid Shape: {protein_manifest.shape}, Atmo Grid Shape: {atom_manifest.shape}")
        return [protein_torch, atom_torch] # 返回處理後的蛋白質和原子網格

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
        atom_manifest = loaded_data['atom_grid'] # 原子網格
        atom_torch = torch.from_numpy(atom_manifest).type(torch.FloatTensor)
        return [protein_torch, atom_torch] # 返回處理後的蛋白質和原子網格

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
    def __init__(self, img_shape=(32, 32, 32), input_dim=1, output_dim=4, embed_dim=768, patch_size=16,
                 num_heads=6, dropout=0.1, ext_layers=[3, 6, 9, 12], norm="instance", base_filters=16,
                 dim_linear_block=3072):
        """
        每一個輸入進來的向量大小為32，因為patch_size為16，所以會被分隔為2*2*2的16*16*16的小塊，所以會有8個patch。
        且每一個patch會被嵌入到大小為768的向量中。
        """
        super().__init__()
        self.num_layers = 12 # Transformer層數
        self.input_dim = input_dim  # 輸入單一維度 
        self.output_dim = output_dim # 輸出4維向量，就是被分類出來的骨架原子
        self.embed_dim = embed_dim # 768
        self.img_shape = img_shape # 輸入資料維度
        self.patch_size = patch_size # 補丁數量
        self.num_heads = num_heads # 注意力機制的
        self.dropout = dropout # 丟棄率為0.1
        self.ext_layers = ext_layers 
        self.patch_dim = [int(x / patch_size) for x in
                          img_shape]

        self.norm = nn.BatchNorm3d if norm == 'batch' else nn.InstanceNorm3d # 選擇歸一化方法

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

            # brown block, final classification
            nn.Conv3d(base_filters, output_dim, kernel_size=1, stride=1))

    def forward(self, x):
        transformer_input = self.embed(x) # 進行3D嵌入
        z3, z6, z9, z12 = map(
            lambda t: rearrange(t, 'b (x y z) d -> b d x y z', x=self.patch_dim[0], y=self.patch_dim[1],
                                z=self.patch_dim[2]), self.transformer(transformer_input)) # Transformer輸出

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

        # concat + yellow conv
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
        return self.out_conv(y) # 返回最終輸出的結果

# 計算每個類別的權重，用於處理類別不平衡問題
def calc_ce_weights(batch):
    y_zeros = (batch == 0.).sum() # 類別 0（無原子）
    y_ones = (batch == 1.).sum() # 類別 1（Cα）
    y_two = (batch == 2.).sum() # 類別 2（N）
    y_three = (batch == 3.).sum() # 類別 3（C）
    
    # 計算每個類別的樣本數量
    nSamples = [y_zeros, y_ones, y_two, y_three]
    
    # 根據每個類別的頻率來計算權重，較少的類別會得到較高的權重
    normedWeights_1 = [1 - (x / sum(nSamples)) for x in nSamples]
    
    # 為了避免除以零的情況，對權重做微小的調整
    normedWeights = [x + 1e-5 for x in normedWeights_1]
    
    # 將權重轉換為 Tensor，並將其放到 GPU 上
    balance_weights = torch.FloatTensor(normedWeights).to("cuda")
    return balance_weights 

# 定義模型類別
class VoxelClassify(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, **model_kwargs):
        super().__init__()
        
        # 保存超參數，便於後續使用
        self.save_hyperparameters()
        # 定義 Transformer_UNET 模型
        self.model = Transformer_UNET(**model_kwargs)
        # 定義交叉熵損失函數
        self.loss_fn = nn.CrossEntropyLoss()

        # 定義評估指標（宏平均和加權平均）
        self.metrics_macro = MetricCollection([Accuracy(task='multiclass', num_classes=4, average='macro', mdmc_average="global"),
                                               Precision(task='multiclass', num_classes=4, average='macro', mdmc_average="global"),
                                               Recall(task='multiclass', num_classes=4, average='macro', mdmc_average="global"),
                                               F1Score(task='multiclass', num_classes=4, average='macro', mdmc_average="global"),
                                               FBetaScore(task='multiclass', num_classes=4, average='macro', mdmc_average="global")])
        
        # 定義加權平均的評估指標
        self.metrics_weighted = MetricCollection([Accuracy(task='multiclass', num_classes=4, average='weighted', mdmc_average="global"),
                                                  Precision(task='multiclass', num_classes=4, average='weighted', mdmc_average="global"),
                                                  Recall(task='multiclass', num_classes=4, average='weighted', mdmc_average="global"),
                                                  F1Score(task='multiclass', num_classes=4, average='weighted', mdmc_average="global"),
                                                  FBetaScore(task='multiclass', num_classes=4, average='weighted', mdmc_average="global")])
        """
        self.metrics_micro = MetricCollection([Accuracy(num_classes=4, average='micro', mdmc_average="global"),
                                               Precision(num_classes=4, average='micro', mdmc_average="global"),
                                               Recall(num_classes=4, average='micro', mdmc_average="global"),
                                               F1Score(num_classes=4, average='micro', mdmc_average="global"),
                                               FBetaScore(num_classes=4, average='micro', mdmc_average="global")])
        """
        
        # 克隆的指標，分別用於訓練、驗證和測試
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

    # 定義前向傳遞的邏輯
    def forward(self, data):
        # 通過 Transformer_UNET 模型進行推理
        x = self.model(data)
        return x

    # 配置優化器和學習率調度器
    def configure_optimizers(self):
        # 使用NAdam優化器
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.hparams.learning_rate)
        
        # 使用ReduceLROnPlateau學習率調度器來根據驗證損失減少學習率
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1,
                                                                  patience=10, eps=1e-10, verbose=True)
        metric_to_track = 'train_loss' # 監控的指標是訓練損失
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': metric_to_track # 監控訓練損失
        }

    # 訓練步驟
    def training_step(self, batch, batch_idx):
        # 從批次中提取蛋白質數據和原子標籤
        protein_data, atom_data = batch[0], batch[1]
        # 將蛋白質數據添加額外的維度，以符合模型要求
        protein_data = torch.unsqueeze(protein_data, 1)
        # 通過模型獲得預測
        y_hat = self.forward(protein_data)
        
        # 計算加權的交叉熵損失
        balance_weights = calc_ce_weights(atom_data) # 計算每個類別的權重
        loss_fn_train = nn.CrossEntropyLoss(weight=balance_weights) # 使用加權的交叉熵損失函數
        loss = loss_fn_train(y_hat, atom_data.long()) # 計算損失
        
        # 計算並記錄評估指標
        metric_log_macro = self.train_metrics_macro(y_hat, atom_data.int())
        # metric_log_micro = self.train_metrics_micro(y_hat, amino_data.int())
        metric_log_weighted = self.train_metrics_weighted(y_hat, atom_data.int())
        
        # 記錄損失和指標
        self.log_dict(metric_log_macro,on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(metric_log_micro)
        self.log_dict(metric_log_weighted, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # 驗證步驟
    def validation_step(self, batch, batch_idx):
        protein_data, atom_data = batch[0], batch[1] # 提取蛋白質數據和原子標籤
        protein_data = torch.unsqueeze(protein_data, 1) # 增加一個維度以符合模型要求
        y_hat = self.forward(protein_data) # 獲得模型預測結果
        loss = self.loss_fn(y_hat, atom_data.long()) # 計算損失
        # loss = loss_fn_train(y_hat, batch.y.long())
        
        # 記錄驗證指標
        metric_log_macro = self.valid_metrics_macro(y_hat, atom_data.int())
        # metric_log_micro = self.valid_metrics_micro(y_hat, amino_data.int())
        metric_log_weighted = self.valid_metrics_weighted(y_hat, atom_data.int())
        self.log_dict(metric_log_macro, on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(metric_log_micro)
        self.log_dict(metric_log_weighted, on_step=True, on_epoch=True, sync_dist=True)
        self.log('valid_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    # 測試步驟
    def test_step(self, batch, batch_idx):
        protein_data, atom_data = batch[0], batch[1] # 提取蛋白質數據和原子標籤
        protein_data = torch.unsqueeze(protein_data, 1) # 增加一個維度以符合模型要求
        y_hat = self.forward(protein_data) # 獲得模型預測結果
        loss = self.loss_fn(y_hat, atom_data.long()) # 計算損失
        # loss = loss_fn_train(y_hat, batch.y.long())
        
        # 記錄測試指標
        metric_log_macro = self.test_metrics_macro(y_hat, atom_data.int())
        # metric_log_micro = self.test_metrics_micro(y_hat, amino_data.int())
        metric_log_weighted = self.test_metrics_weighted(y_hat, atom_data.int())
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

# 訓練模型的主函數
def train_node_classifier():
    pl.seed_everything(42) # 設置隨機種子以確保實驗可重現
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
    # args.resume_from_checkpoint = "CHECKPOINT_PATH/atom-epoch=02-valid_loss=0.192797.ckpt" 

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

    model = VoxelClassify(learning_rate=1e-4, img_shape=(32, 32, 32), input_dim=1, output_dim=4, embed_dim=768,
                          patch_size=16,
                          num_heads=6, dropout=0.1, ext_layers=[3, 6, 9, 12], norm="instance", base_filters=16,
                          dim_linear_block=3072)
    
    # 計算模型的參數數量並顯示
    print("Model's trainable parameters", count_parameters(model))
    
    # 使用PyTorch Lightning的trainer來訓練模型
    trainer = pl.Trainer.from_argparse_args(args)

    # early_stopping_callback = EarlyStopping(monitor='valid_loss', mode='min', min_delta=0.0, patience=5)
    # 使用Weights and Biases來記錄訓練過程
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_top_k=5, dirpath=args.save_dir,
                                          filename='atom-{epoch:02d}-{valid_loss:.6f}')
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
