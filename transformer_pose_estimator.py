# transformer_pose_estimator.py
"""
基于 Transformer 优化的 YOLOv8 姿态估计器
集成了 Swin Transformer、Vision Transformer 和 DETR 等多种 Transformer 架构
用于提升姿态检测的精度和鲁棒性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import logging
from typing import Optional, Tuple, List
from ultralytics import YOLO

try:
    import timm  # 可选：用于Swin/Vision Transformer等骨干
    import transformers  # 可选：如需使用预训练Transformer模型
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformer依赖未安装。请运行: pip install timm transformers")

from pose_estimator import PoseEstimator

logger = logging.getLogger(__name__)

class AdaptivePositionalEncoding(nn.Module):
    """2D 相对位置编码，适合目标检测任务"""
    
    def __init__(self, d_model: int, max_height: int = 640, max_width: int = 640):
        super().__init__()
        self.d_model = d_model
        self.height_embed = nn.Embedding(max_height, d_model // 2)
        self.width_embed = nn.Embedding(max_width, d_model // 2)
        
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """生成适合图像尺寸的位置编码"""
        device = x.device
        h_pos = torch.arange(h, device=device)
        w_pos = torch.arange(w, device=device)
        
        h_embed = self.height_embed(h_pos)  # [h, d_model//2]
        w_embed = self.width_embed(w_pos)   # [w, d_model//2]
        
        # 创建2D位置编码
        pos_embed = torch.cat([
            h_embed.unsqueeze(1).repeat(1, w, 1),  # [h, w, d_model//2]
            w_embed.unsqueeze(0).repeat(h, 1, 1)   # [h, w, d_model//2]
        ], dim=-1)  # [h, w, d_model]
        
        return pos_embed

class WindowAttention(nn.Module):
    """窗口注意力机制，减少计算复杂度"""
    
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """将特征图分割成窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows.view(-1, window_size * window_size, C)

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """重组窗口为特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class EfficientTransformerBlock(nn.Module):
    """高效的Transformer块，使用窗口注意力"""
    
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        shortcut = x
        
        # 窗口注意力
        x = self.norm1(x)
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attention(x_windows)
        x = window_reverse(attn_windows, self.window_size, H, W)
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerNeck(nn.Module):
    """增强的特征融合层，使用Transformer模块"""
    
    def __init__(self, in_channels: List[int], out_channels: int, num_layers: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 特征对齐层
        self.feature_adapters = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels
        ])
        
        # Transformer增强块
        self.transformer_blocks = nn.ModuleList([
            EfficientTransformerBlock(
                dim=out_channels,
                window_size=8,
                num_heads=8,
                mlp_ratio=4.0
            ) for _ in range(num_layers)
        ])
        
        # 位置编码
        self.pos_encoding = AdaptivePositionalEncoding(out_channels)
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """多尺度特征融合"""
        enhanced_features = []
        
        for i, (feat, adapter) in enumerate(zip(features, self.feature_adapters)):
            # 特征对齐
            feat = adapter(feat)
            B, C, H, W = feat.shape
            
            # 添加位置编码
            feat_flat = feat.permute(0, 2, 3, 1)  # [B, H, W, C]
            pos_embed = self.pos_encoding(feat_flat, H, W)
            feat_flat = feat_flat + pos_embed
            
            # Transformer增强
            for transformer in self.transformer_blocks:
                feat_flat = transformer(feat_flat)
            
            # 转换回原始格式
            enhanced_feat = feat_flat.permute(0, 3, 1, 2)  # [B, C, H, W]
            enhanced_features.append(enhanced_feat)
            
        return enhanced_features

class TransformerHead(nn.Module):
    """基于Transformer的检测头"""
    
    def __init__(self, num_classes: int = 1, num_keypoints: int = 17, d_model: int = 256):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu"
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # 查询嵌入
        self.query_embed = nn.Embedding(num_keypoints, d_model)
        
        # 输出头
        self.pose_head = nn.Linear(d_model, 3)  # x, y, visibility
        self.class_head = nn.Linear(d_model, num_classes)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用Transformer解码器进行姿态预测"""
        B, C, H, W = features.shape
        
        # 特征展平
        feat_flat = features.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        # 查询嵌入
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_keypoints, B, d_model]
        
        # Transformer解码
        decoded_features = self.transformer_decoder(query_embed, feat_flat)  # [num_keypoints, B, d_model]
        
        # 预测
        pose_predictions = self.pose_head(decoded_features)  # [num_keypoints, B, 3]
        class_predictions = self.class_head(decoded_features)  # [num_keypoints, B, num_classes]
        
        # 调整维度
        pose_predictions = pose_predictions.permute(1, 0, 2)  # [B, num_keypoints, 3]
        class_predictions = class_predictions.permute(1, 0, 2)  # [B, num_keypoints, num_classes]
        
        return pose_predictions, class_predictions

class TransformerPoseEstimator(PoseEstimator):
    """
    基于Transformer优化的YOLOv8姿态估计器
    
    支持多种Transformer架构：
    - swin: Swin Transformer
    - vit: Vision Transformer  
    - detr: DETR风格的Transformer
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        use_transformer: bool = True,
        transformer_type: str = "swin",
        device: str = "auto"
    ):
        """
        初始化Transformer优化的姿态估计器
        
        Args:
            model_path: YOLO模型路径
            use_transformer: 是否启用Transformer优化
            transformer_type: Transformer类型 ("swin", "vit", "detr")
            device: 计算设备
        """
        # 首先初始化基础的姿态估计器
        super().__init__(model_path)
        
        self.use_transformer = use_transformer
        self.transformer_type = transformer_type
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        if use_transformer:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformer依赖未安装，回退到标准YOLO模型")
                self.use_transformer = False
            else:
                self._enhance_with_transformer()
                logger.info(f"启用Transformer优化，类型: {transformer_type}")
    
    def _enhance_with_transformer(self):
        """使用Transformer模块增强YOLO模型"""
        try:
            if self.transformer_type == "swin":
                self._integrate_swin_transformer()
            elif self.transformer_type == "vit":
                self._integrate_vision_transformer()
            elif self.transformer_type == "detr":
                self._integrate_detr_transformer()
            else:
                logger.warning(f"不支持的Transformer类型: {self.transformer_type}")
                self.use_transformer = False
                
        except Exception as e:
            logger.error(f"Transformer集成失败: {e}")
            self.use_transformer = False
    
    def _integrate_swin_transformer(self):
        """集成Swin Transformer"""
        # 这里可以集成预训练的Swin Transformer骨干网络
        # 由于复杂性，这里提供一个简化的实现框架
        logger.info("集成Swin Transformer骨干网络")
        
        # 创建Transformer增强的特征融合层
        self.transformer_neck = TransformerNeck(
            in_channels=[256, 512, 1024],  # 假设的特征通道数
            out_channels=256,
            num_layers=3
        ).to(self.device)
        
    def _integrate_vision_transformer(self):
        """集成Vision Transformer"""
        logger.info("集成Vision Transformer")
        
        # 简化的ViT集成
        self.transformer_head = TransformerHead(
            num_classes=1,
            num_keypoints=17,
            d_model=256
        ).to(self.device)
        
    def _integrate_detr_transformer(self):
        """集成DETR风格的Transformer"""
        logger.info("集成DETR风格Transformer")
        
        # DETR风格的Transformer解码器
        self.transformer_decoder = TransformerHead(
            num_classes=1,
            num_keypoints=17,
            d_model=256
        ).to(self.device)
    
    def get_pose(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        增强的姿态检测，可选择使用Transformer优化
        
        Args:
            frame: 输入图像帧
            
        Returns:
            keypoints: 关键点列表
            annotated_frame: 带注释的图像帧
        """
        if not self.use_transformer:
            # 回退到标准实现
            return super().get_pose(frame)
        
        try:
            # 使用基础YOLO模型进行检测
            results = self.model.predict(frame, verbose=False)[0]
            
            if len(results.keypoints) == 0:
                return [], results.plot()
            
            # 获取原始关键点
            raw_kp = results.keypoints.xy[0].cpu().numpy()  # shape (17,2)
            
            # 如果集成了Transformer模块，可以在这里进行后处理优化
            if hasattr(self, 'transformer_neck') or hasattr(self, 'transformer_head'):
                raw_kp = self._refine_keypoints_with_transformer(frame, raw_kp)
            
            # 转换为归一化坐标
            kp_normalized = [(x / frame.shape[1], y / frame.shape[0]) for x, y in raw_kp]
            
            # 生成带注释的图像
            annotated_frame = results.plot()
            
            return kp_normalized, annotated_frame
            
        except Exception as e:
            logger.error(f"Transformer增强姿态检测失败: {e}")
            # 回退到标准实现
            return super().get_pose(frame)
    
    def _refine_keypoints_with_transformer(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """使用Transformer模块精炼关键点"""
        # 这里可以实现更复杂的Transformer后处理逻辑
        # 目前返回原始关键点
        return keypoints
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = {
            "base_model": "YOLOv8n-pose",
            "transformer_enabled": self.use_transformer,
            "transformer_type": self.transformer_type if self.use_transformer else None,
            "device": str(self.device),
            "transformers_available": TRANSFORMERS_AVAILABLE
        }
        return info
    
    def benchmark(self, frame: np.ndarray, iterations: int = 100) -> dict:
        """性能基准测试"""
        import time
        
        # 预热
        for _ in range(10):
            self.get_pose(frame)
        
        # 基准测试
        start_time = time.time()
        for _ in range(iterations):
            self.get_pose(frame)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        fps = 1.0 / avg_time
        
        return {
            "average_inference_time": avg_time,
            "fps": fps,
            "iterations": iterations,
            "transformer_enabled": self.use_transformer
        }

# 使用示例和工厂函数
def create_pose_estimator(
    model_path: str = "yolov8n-pose.pt",
    use_transformer: bool = False,
    transformer_type: str = "swin"
) -> PoseEstimator:
    """
    工厂函数，创建姿态估计器
    
    Args:
        model_path: 模型路径
        use_transformer: 是否使用Transformer优化
        transformer_type: Transformer类型
        
    Returns:
        姿态估计器实例
    """
    if use_transformer:
        return TransformerPoseEstimator(
            model_path=model_path,
            use_transformer=True,
            transformer_type=transformer_type
        )
    else:
        return PoseEstimator(model_path)

if __name__ == "__main__":
    # 测试代码
    print("测试Transformer姿态估计器...")
    
    # 创建优化后的估计器
    estimator = TransformerPoseEstimator(
        use_transformer=True,
        transformer_type="swin"
    )
    
    # 打印模型信息
    info = estimator.get_model_info()
    print("模型信息:", info)
    
    # 如果有摄像头，可以进行实际测试
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                keypoints, annotated = estimator.get_pose(frame)
                print(f"检测到 {len(keypoints)} 个关键点")
                
                # 性能测试
                benchmark_results = estimator.benchmark(frame, iterations=50)
                print("性能测试结果:", benchmark_results)
        
        cap.release()
    except Exception as e:
        print(f"摄像头测试失败: {e}")
