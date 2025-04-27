# 导入 PyTorch 相关库
import torch # PyTorch 深度学习框架核心库
import torchvision.models as models # 包含预训练模型的模块
import torchvision.transforms as transforms # 提供常用图像预处理操作的模块
from PIL import Image # Python Imaging Library (Pillow)，用于图像文件操作

# --- 模型加载与配置 ---
# 加载预训练的 ResNet-18 模型
# pretrained=True 表示加载在 ImageNet 数据集上预训练过的权重
# 注意：`pretrained` 参数在较新版本 torchvision 中已弃用，推荐使用 `weights` 参数，
# 例如 `models.resnet18(weights=models.ResNet18_Weights.DEFAULT)`
model = models.resnet18(pretrained=True)

# 修改模型结构以用于特征提取
# ResNet-18 的原始结构包含一个最后的线性层 (全连接层) 用于分类
# 为了提取特征向量，我们移除这个最后的线性层
# `model.children()` 返回模型的所有直接子模块
# `[:-1]` 表示选取除了最后一个元素之外的所有子模块
# `torch.nn.Sequential` 将这些子模块按顺序组合成一个新的序列模型
model = torch.nn.Sequential(*list(model.children())[:-1])

# 将模型设置为评估模式 (evaluation mode)
# 这会关闭 Dropout 和 Batch Normalization 的更新，确保推理结果的一致性
model.eval()

# --- 图像预处理流程定义 ---
# 定义一个图像预处理的转换序列 (Compose)
preprocess = transforms.Compose([
    # 1. 调整图像大小：将图像短边调整到 256 像素，长边按比例缩放
    transforms.Resize(256),
    # 2. 中心裁剪：从图像中心裁剪出 224x224 大小的区域
    transforms.CenterCrop(224),
    # 3. 转换为 Tensor：将 PIL 图像对象转换为 PyTorch Tensor 对象
    #    像素值会被归一化到 [0.0, 1.0] 范围
    transforms.ToTensor(),
    # 4. 标准化：使用 ImageNet 数据集的均值和标准差对 Tensor 进行标准化
    #    这是预训练模型要求的标准预处理步骤
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 特征提取函数 ---
def extract_features(image_path):
    """
    使用预训练的 ResNet-18 模型从给定图像中提取特征向量。

    参数:
        image_path (str): 图像文件的路径。

    返回:
        numpy.ndarray: 经过 L2 归一化的 512 维特征向量。
    """
    # 打开图像文件，并确保转换为 RGB 格式 (有些图像可能是灰度或 RGBA)
    image = Image.open(image_path).convert('RGB')
    # 应用前面定义的预处理流程，并将结果增加一个批次维度 (unsqueeze(0))
    # 模型期望输入是 4D Tensor: (batch_size, channels, height, width)
    image = preprocess(image).unsqueeze(0)

    # 使用 torch.no_grad() 上下文管理器，禁用梯度计算
    # 在推理阶段不需要计算梯度，可以节省内存并加速计算
    with torch.no_grad():
        # 将预处理后的图像输入模型，得到输出特征
        features = model(image)

    # 处理模型输出
    # `squeeze()` 移除所有维度为 1 的维度 (例如批次维度)
    # `flatten()` 将多维特征图展平成一维向量
    features = features.squeeze().flatten()

    # 对提取到的特征向量进行 L2 归一化
    # L2 归一化使得向量长度为 1，这对于使用 L2 距离进行相似度比较通常是有益的
    # `torch.nn.functional.normalize` 函数执行归一化操作
    # p=2 表示 L2 范数，dim=0 表示在第 0 维 (即向量本身) 上进行归一化
    features_norm = torch.nn.functional.normalize(features, p=2, dim=0)

    # 将 PyTorch Tensor 转换为 NumPy 数组并返回
    # .numpy() 方法用于转换
    return features_norm.numpy()
