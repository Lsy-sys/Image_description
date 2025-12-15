"""
Faster R-CNN 区域特征提取器
用于提取图像的多个区域特征，供Transformer模型使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align, nms
from PIL import Image
import numpy as np


class FasterRCNNFeatureExtractor:
    """
    Faster R-CNN 区域特征提取器
    使用预训练的 Faster R-CNN 模型提取图像中的区域特征
    """
    
    def __init__(self, device, min_score=0.05, max_regions=36, use_nms=True):
        """
        Args:
            device: 运行设备 (cuda/cpu)
            min_score: 最小置信度阈值
            max_regions: 最大区域数量
            use_nms: 是否使用非极大值抑制
        """
        self.device = device
        self.min_score = min_score
        self.max_regions = max_regions
        self.use_nms = use_nms
        
        # 加载预训练的 Faster R-CNN 模型
        print("加载Faster R-CNN模型...")
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(device)
        
        # 获取backbone用于特征提取
        self.backbone = self.model.backbone
        
        # 特征维度（FPN的最后一层通常是256维，但我们需要投影到2048）
        self.feature_dim = 256
        
        # 创建特征投影层（从256维投影到2048维）
        self.projection = nn.Linear(256, 2048).to(device)
        nn.init.xavier_uniform_(self.projection.weight)
        self.projection.eval()
        
        print("Faster R-CNN模型加载完成")
    
    def extract_roi_features(self, feature_maps, boxes, image_size):
        """
        使用RoI Align从特征图中提取区域特征
        Args:
            feature_maps: FPN特征图字典
            boxes: 边界框 (N, 4) 格式为 (x1, y1, x2, y2)
            image_size: 图像尺寸 (height, width)
        Returns:
            区域特征 (N, 256, 7, 7) -> (N, 256)
        """
        # 使用p3特征图（中等分辨率，通常效果最好）
        # 或者根据boxes大小自动选择最合适的特征层
        if '1' in feature_maps:
            feature_map = feature_maps['1']  # p3
        elif '2' in feature_maps:
            feature_map = feature_maps['2']  # p4
        else:
            feature_map = list(feature_maps.values())[-1]  # 使用最后一层
        
        # 准备boxes格式：[batch_idx, x1, y1, x2, y2]
        batch_boxes = []
        for i, box in enumerate(boxes):
            batch_boxes.append([0, box[0].item(), box[1].item(), box[2].item(), box[3].item()])
        batch_boxes = torch.tensor(batch_boxes, dtype=torch.float32, device=self.device)
        
        # 使用RoI Align提取特征
        # output_size: 输出的特征图大小
        # spatial_scale: 特征图相对于原图的缩放比例
        h, w = image_size
        feat_h, feat_w = feature_map.shape[-2:]
        spatial_scale = feat_h / h
        
        try:
            roi_features = roi_align(
                feature_map,
                batch_boxes,
                output_size=(7, 7),
                spatial_scale=spatial_scale,
                sampling_ratio=2
            )
            # 全局平均池化
            roi_features = F.adaptive_avg_pool2d(roi_features, (1, 1))
            roi_features = roi_features.squeeze(-1).squeeze(-1)  # (N, 256)
            return roi_features
        except Exception as e:
            print(f"RoI Align失败: {e}，使用简化方法")
            # 如果RoI Align失败，使用简化方法
            return self._extract_features_simple(feature_map, boxes, (h, w))
    
    def _extract_features_simple(self, feature_map, boxes, image_size):
        """简化的特征提取方法（当RoI Align不可用时）"""
        h, w = image_size
        feat_h, feat_w = feature_map.shape[-2:]
        region_features = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.cpu().numpy()
            # 映射到特征图坐标
            fx1 = max(0, int(x1 * feat_w / w))
            fy1 = max(0, int(y1 * feat_h / h))
            fx2 = min(feat_w, int(x2 * feat_w / w))
            fy2 = min(feat_h, int(y2 * feat_h / h))
            
            if fx2 > fx1 and fy2 > fy1:
                region = feature_map[:, :, fy1:fy2, fx1:fx2]
                region_feat = F.adaptive_avg_pool2d(region, (1, 1))
                region_feat = region_feat.squeeze(-1).squeeze(-1)
                region_features.append(region_feat)
        
        if len(region_features) > 0:
            return torch.stack(region_features, dim=0)
        else:
            global_feat = F.adaptive_avg_pool2d(feature_map, (1, 1))
            global_feat = global_feat.squeeze(-1).squeeze(-1)
            return global_feat.unsqueeze(0)
    
    def extract_regions(self, image_tensor, num_regions=None):
        """
        提取图像的区域特征
        Args:
            image_tensor: 图像张量 (batch_size, 3, H, W) 或 (3, H, W)
            num_regions: 期望的区域数量（如果不指定，使用max_regions）
        Returns:
            区域特征 (batch_size, num_regions, feature_dim) 或 (1, num_regions, feature_dim)
        """
        if num_regions is None:
            num_regions = self.max_regions
        
        # 处理批次维度
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = image_tensor.shape[0]
        batch_features = []
        
        for i in range(batch_size):
            single_image = image_tensor[i:i+1]
            
            with torch.no_grad():
                # 第一步：运行Faster R-CNN检测，获取边界框
                detections = self.model(single_image)
                
                boxes = detections[0]['boxes']
                scores = detections[0]['scores']
                
                # 过滤低置信度的检测
                valid_indices = scores >= self.min_score
                boxes = boxes[valid_indices]
                scores = scores[valid_indices]
                
                # 非极大值抑制（NMS）去除重叠的检测框
                if self.use_nms and len(boxes) > 0:
                    keep = nms(boxes, scores, iou_threshold=0.5)
                    boxes = boxes[keep]
                    scores = scores[keep]
                
                # 获取特征图
                features_dict = self.backbone(single_image)
                
                # 按分数排序，选择top-k个区域
                if len(boxes) > 0:
                    sorted_indices = torch.argsort(scores, descending=True)
                    boxes = boxes[sorted_indices[:num_regions]]
                    
                    # 使用RoI Align提取区域特征
                    img_h, img_w = single_image.shape[-2:]
                    region_features = self.extract_roi_features(
                        features_dict, boxes, (img_h, img_w)
                    )
                    
                    # 如果提取的区域少于期望数量，用全局特征填充
                    num_extracted = region_features.shape[0]
                    if num_extracted < num_regions:
                        # 获取全局特征
                        global_feat_map = features_dict.get('3', list(features_dict.values())[-1])
                        global_feat = F.adaptive_avg_pool2d(global_feat_map, (1, 1))
                        global_feat = global_feat.squeeze(-1).squeeze(-1)  # (256,)
                        
                        # 填充
                        padding = global_feat.unsqueeze(0).repeat(num_regions - num_extracted, 1)
                        region_features = torch.cat([region_features, padding], dim=0)
                    
                    region_features = region_features[:num_regions]
                else:
                    # 如果没有检测到区域，使用全局特征
                    global_feat_map = features_dict.get('3', list(features_dict.values())[-1])
                    global_feat = F.adaptive_avg_pool2d(global_feat_map, (1, 1))
                    global_feat = global_feat.squeeze(-1).squeeze(-1)  # (256,)
                    region_features = global_feat.unsqueeze(0).repeat(num_regions, 1)
            
            # 投影到2048维
            with torch.no_grad():
                region_features = self.projection(region_features)  # (num_regions, 2048)
            
            batch_features.append(region_features)
        
        result = torch.stack(batch_features, dim=0)  # (batch_size, num_regions, 2048)
        
        # 如果输入是单张图像，移除批次维度
        if squeeze_output:
            result = result.squeeze(0)
        
        return result
    
    def extract_regions_simple(self, image_tensor, num_regions=36):
        """
        简化版本：使用均匀网格采样区域特征
        当Faster R-CNN检测失败时使用此方法作为备选
        Args:
            image_tensor: 图像张量 (1, 3, H, W)
            num_regions: 区域数量
        Returns:
            区域特征 (1, num_regions, 2048)
        """
        with torch.no_grad():
            # 获取特征图
            features_dict = self.backbone(image_tensor)
            feature_map = features_dict.get('3', features_dict.get('2'))
            
            # 使用均匀网格采样
            h, w = feature_map.shape[-2:]
            grid_h = int(np.sqrt(num_regions))
            grid_w = int(np.ceil(num_regions / grid_h))
            
            region_features = []
            step_h = h // grid_h
            step_w = w // grid_w
            
            for i in range(grid_h):
                for j in range(grid_w):
                    if len(region_features) >= num_regions:
                        break
                    y1 = i * step_h
                    y2 = min((i + 1) * step_h, h)
                    x1 = j * step_w
                    x2 = min((j + 1) * step_w, w)
                    
                    region = feature_map[:, :, y1:y2, x1:x2]
                    region_feat = torch.nn.functional.adaptive_avg_pool2d(region, (1, 1))
                    region_feat = region_feat.squeeze(-1).squeeze(-1)
                    region_features.append(region_feat)
            
            # 如果区域数量不足，用最后一个区域填充
            while len(region_features) < num_regions:
                region_features.append(region_features[-1])
            
            region_features = torch.stack(region_features[:num_regions], dim=0)
            
            # 投影到2048维
            if region_features.shape[-1] == 256:
                if not hasattr(self, 'projection'):
                    self.projection = nn.Linear(256, 2048).to(self.device)
                    nn.init.xavier_uniform_(self.projection.weight)
                region_features = self.projection(region_features)
            
            return region_features.unsqueeze(0)  # (1, num_regions, 2048)

