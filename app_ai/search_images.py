# 导入所需的库
from pymilvus import connections, Collection # Milvus 客户端库，用于连接和操作集合
import numpy as np # 用于数值计算 (虽然在此脚本中未直接使用，但通常与向量操作相关)
from resnet import extract_features # 从自定义的 resnet 模块导入特征提取函数

# --- Milvus 连接配置 ---
# 使用别名 "default" 连接到本地运行的 Milvus 实例
connections.connect(
    alias="default", # 连接别名
    host='localhost', # Milvus 服务器地址
    port='19530' # Milvus 服务器端口
)

# --- Milvus 集合配置 ---
# 定义要操作的集合名称 (应与 insert_images.py 中使用的名称一致)
collection_name = "intangible_cultural_heritage_images"

# --- 加载 Milvus 集合 ---
# 尝试获取并加载指定名称的集合
try:
    # 获取集合对象
    collection = Collection(name=collection_name)
    # 将集合加载到内存中以进行搜索操作
    collection.load()
    # 打印加载成功的提示信息
    print(f"集合 {collection_name} 加载成功")
# 捕获加载过程中可能发生的任何异常
except Exception as e:
    # 如果加载失败，打印错误信息和用户提示
    print(f"加载集合 {collection_name} 时出错: {e}")
    print("请确保集合存在、已创建索引且 Milvus 服务器正在运行。")
    print("您可能需要先运行 insert_images.py 脚本来创建集合并插入数据。")
    # 退出脚本，因为无法继续执行搜索
    exit()

# --- 相似度搜索函数 ---
def search_similar_vectors(query_vector, top_k=10):
    """
    在 Milvus 集合中搜索与给定查询向量最相似的 top_k 个向量。

    参数:
        query_vector (numpy.ndarray): 用于查询的特征向量 (应与集合中存储的向量维度相同)。
        top_k (int): 希望返回的最相似结果的数量，默认为 10。

    返回:
        list: 一个包含相似结果字典的列表。每个字典包含 'id' (Milvus 中的实体 ID)
              和 'distance' (与查询向量的 L2 距离)。列表按距离升序排列。
    """
    # 定义搜索参数
    search_params = {
        "metric_type": "L2", # 使用 L2 距离作为相似度度量 (应与创建索引时一致)
        "params": {"nprobe": 10} # 搜索参数，nprobe 控制搜索时查找的聚类数量，影响召回率和性能
                                 # nprobe 的值通常需要根据数据集大小和性能要求进行调整
    }
    # 执行搜索操作
    results = collection.search(
        data=[query_vector], # 查询向量列表 (这里只有一个查询向量)
        anns_field="embedding", # 指定在哪一个向量字段上进行搜索
        param=search_params, # 搜索参数
        limit=top_k, # 返回结果的数量上限
        # 指定需要从搜索结果中额外获取的字段 (除了 id 和 distance)
        # 我们需要获取存储在 Milvus 中的 image_filename
        output_fields=["id", "image_filename"] # 请求返回 id 和 image_filename
    )

    # --- 格式化搜索结果 ---
    formatted_results = [] # 初始化用于存储格式化结果的列表
    # Milvus 的 search 方法返回一个列表，每个元素对应一个查询向量的结果
    # 因为我们只有一个查询向量，所以只关心 results[0]
    if results and results[0]: # 检查 results 是否非空且第一个查询有命中结果 (hits)
        # 遍历第一个查询的所有命中结果 (hits)
        for hit in results[0]:
            # 将每个命中结果的 id, distance 和 filename 提取出来，存入字典
            # hit.entity.get('field_name') 用于获取 output_fields 中指定的字段值
            filename = hit.entity.get('image_filename', '未知文件名') # 提供默认值以防万一
            formatted_results.append({
                'id': hit.id, # 命中向量在 Milvus 中的 ID
                'distance': hit.distance, # 命中向量与查询向量的距离
                'filename': filename # 获取到的图像文件名
            })
    # 返回格式化后的结果列表
    return formatted_results

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 查询参数配置 ---
    # 设置要查询的图片路径 (请根据实际情况修改为你本地的图片路径)
    image_path = "D:\\Code\\heritage\\app_ai\\test3.png"
    # 设置希望返回的最相似图片的数量
    top_k = 5

    # --- 执行查询与结果展示 ---
    try:
        # 打印开始提取特征的提示
        print(f"正在提取查询图片 {image_path} 的特征...")
        # 调用 resnet 模块的 extract_features 函数提取查询图片的特征向量
        query_vector = extract_features(image_path)

        # 打印开始搜索的提示
        print(f"正在搜索前 {top_k} 个相似图像...")
        # 调用 search_similar_vectors 函数执行相似度搜索
        similar_results = search_similar_vectors(query_vector, top_k=top_k)

        # 检查是否找到了相似结果
        if similar_results:
            # 如果找到结果，打印表头
            print("\n相似图像搜索结果:")
            # 遍历格式化后的搜索结果列表
            for idx, result in enumerate(similar_results, 1):
                # 计算一个简单的相似度百分比 (基于 L2 距离，仅为示例)
                similarity_percent = max(0, (1 - result['distance'])) * 100 # 确保百分比不为负
                # 打印每个相似结果的信息：排名、ID、文件名、距离和计算出的相似度百分比
                print(f"第{idx}个相似图像 - ID: {result['id']}, 文件名: {result['filename']}, 距离: {result['distance']:.4f}, 相似度: {similarity_percent:.2f}%")
        else:
            # 如果没有找到相似结果，打印提示信息
            print("未找到相似图像。")

    # --- 异常处理 ---
    # 捕获文件未找到的错误 (如果查询图片路径无效)
    except FileNotFoundError:
        print(f"错误：未找到查询图片文件：{image_path}")
    # 捕获其他可能发生的异常 (如 Milvus 连接问题、特征提取失败等)
    except Exception as e:
        print(f"搜索过程中发生错误：{e}")
