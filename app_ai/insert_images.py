# 导入所需的库
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility  # Milvus 客户端库
import numpy as np  # 用于数值计算
from resnet import extract_features  # 从自定义的 resnet 模块导入特征提取函数
import os  # 用于操作系统相关操作，如路径处理
import hashlib  # 用于计算文件哈希值

# --- Milvus 连接配置 ---
# 使用别名 "default" 连接到本地运行的 Milvus 实例
connections.connect(
    alias="default",  # 连接别名
    host='192.168.1.100',  # Milvus 服务器地址
    port='19530'  # Milvus 服务器端口
)

# --- Milvus 集合 Schema 定义 ---
# 定义集合中每个字段的模式
fields = [
    # 主键字段：INT64 类型，自动生成 ID
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True,
                auto_id=True),
    # 嵌入向量字段：FLOAT_VECTOR 类型，维度为 512 (由 ResNet18 模型决定)
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    # 图像文件名字段：VARCHAR 类型，最大长度 255
    FieldSchema(name="image_filename", dtype=DataType.VARCHAR, max_length=255),
    # 图像内容哈希值字段：VARCHAR 类型，最大长度 64 (MD5 哈希长度为 32，这里设为 64 足够)
    FieldSchema(name="image_hash", dtype=DataType.VARCHAR, max_length=64)
]
# 创建集合 Schema 对象，包含字段定义和描述信息
schema = CollectionSchema(fields=fields, description="非遗图像特征向量集合 (基于文件名和哈希去重)")
# 定义集合名称
collection_name = "intangible_cultural_heritage_images"

# --- 初始化或获取 Milvus 集合对象 ---
collection = None  # 先将集合对象初始化为 None
# 检查指定名称的集合是否存在
if not utility.has_collection(collection_name):
    # 如果集合不存在，打印提示信息并创建新集合
    print(f"集合 {collection_name} 不存在，正在创建...")
    collection = Collection(name=collection_name,
                            schema=schema)  # 使用定义的 Schema 创建集合
    # --- 为新创建的集合创建索引 ---
    print(f"为新集合 {collection_name} 创建索引...")
    # 定义索引参数：使用 L2 距离度量，索引类型为 IVF_FLAT，聚类数量为 1024
    index_params = {
        "metric_type": "L2",  # 距离度量类型
        "index_type": "IVF_FLAT",  # 索引类型
        "params": {
            "nlist": 1024
        }  # 索引参数，nlist 是聚类中心的数量
    }
    # 在 "embedding" 字段上创建索引
    collection.create_index(
        field_name="embedding",  # 需要创建索引的字段名
        index_params=index_params  # 索引参数
    )
    print(f"已为新集合创建索引 {index_params['index_type']}")  # 打印索引创建成功的提示
    # --- 创建索引结束 ---
else:
    # 如果集合已存在，获取现有集合对象
    collection = Collection(name=collection_name)
    print(f"集合 {collection_name} 已存在，直接使用")
    # --- 检查现有集合是否有索引 ---
    # 如果现有集合没有索引
    if not collection.has_index():
        # 打印警告信息并创建索引
        print(f"警告：集合 {collection_name} 存在但没有索引，正在创建...")
        # 定义索引参数 (同上)
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": 1024
            }
        }
        # 在 "embedding" 字段上创建索引
        collection.create_index(field_name="embedding",
                                index_params=index_params)
        print(f"已为现有集合创建索引 {index_params['index_type']}")  # 打印索引创建成功的提示
    # --- 索引检查结束 ---

# --- 检查集合是否为空 ---
# 确保 collection 对象已成功初始化后再检查实体数量
if collection and collection.num_entities > 0:
    # 如果集合中已有数据，打印警告信息
    print(f"警告：集合中已有 {collection.num_entities} 条数据，继续操作将会追加新数据")

# --- 全局函数定义 ---


# 计算图像文件的 MD5 哈希值
def calculate_image_hash(image_path):
    """
    计算给定图像文件的 MD5 哈希值。

    参数:
        image_path (str): 图像文件的完整路径。

    返回:
        str: 图像内容的 MD5 哈希值 (十六进制字符串)。
    """
    # 以二进制读取模式打开文件
    with open(image_path, 'rb') as f:
        # 读取文件所有内容并计算 MD5 哈希值
        file_hash = hashlib.md5(f.read()).hexdigest()
    # 返回计算得到的哈希值
    return file_hash


# 检查图像是否已存在于 Milvus 集合中 (基于文件名和哈希值)
def is_image_exists(image_filename, image_hash):
    """
    查询 Milvus 集合，检查具有相同文件名或相同哈希值的图像是否已存在。

    参数:
        image_filename (str): 图像的文件名。
        image_hash (str): 图像内容的 MD5 哈希值。

    返回:
        bool: 如果图像已存在则返回 True，否则返回 False。
    """
    # 构建查询表达式：查找 image_filename 匹配或 image_hash 匹配的记录
    # 注意：Milvus 查询表达式中的字符串值需要用双引号括起来
    expr = f'image_filename == "{image_filename}" or image_hash == "{image_hash}"'
    # 执行查询，指定查询表达式和需要输出的字段
    results = collection.query(
        expr=expr,  # 查询条件表达式
        output_fields=["id", "image_filename"]  # 指定返回结果中包含的字段
    )
    # 如果查询结果列表不为空 (即找到匹配记录)，则表示图像已存在
    return len(results) > 0


# 向 Milvus 集合插入图像特征向量、文件名和哈希值
def insert_vectors(vectors, image_paths):
    """
    将图像特征向量、文件名和哈希值批量插入到 Milvus 集合中，
    并在插入前检查重复项。

    参数:
        vectors (list): 包含图像特征向量 (numpy 数组或列表) 的列表。
        image_paths (list): 包含对应图像文件完整路径的列表。
    """
    # 检查输入的向量列表和路径列表长度是否一致
    if len(vectors) != len(image_paths):
        print("错误：向量数量与图像路径数量不匹配")
        return  # 如果不匹配则直接返回

    # --- 准备插入数据 ---
    # 将 numpy 数组转换为列表 (如果需要)，因为 Milvus Python SDK 通常接受列表格式
    embeddings = [v.tolist() for v in vectors] if isinstance(
        vectors[0], np.ndarray) else vectors
    # 从完整路径中提取文件名
    image_filenames = [os.path.basename(path) for path in image_paths]
    # 计算每个图像文件的哈希值
    image_hashes = [calculate_image_hash(path) for path in image_paths]

    # --- 检查重复并筛选需要插入的数据 ---
    new_embeddings = []  # 存储新的特征向量
    new_filenames = []  # 存储新的文件名
    new_hashes = []  # 存储新的哈希值
    skipped_count = 0  # 记录跳过的重复图像数量

    # 遍历每个待处理的图像信息
    for i, (embedding, filename, hash_value) in enumerate(
            zip(embeddings, image_filenames, image_hashes)):
        # 调用 is_image_exists 函数检查图像是否已存在
        if is_image_exists(filename, hash_value):
            # 如果已存在，打印跳过信息并增加计数器
            print(f"跳过已存在的图像: {filename}")
            skipped_count += 1
        else:
            # 如果是新图像，将其信息添加到待插入列表中
            new_embeddings.append(embedding)
            new_filenames.append(filename)
            new_hashes.append(hash_value)

    # --- 执行插入操作 ---
    # 如果存在需要插入的新图像数据
    if new_embeddings:
        # 准备插入的数据列表，顺序与 Schema 定义一致
        data_to_insert = [
            new_embeddings,  # 特征向量字段数据
            new_filenames,  # 图像文件名字段数据
            new_hashes  # 图像哈希字段数据
        ]
        # 调用 collection.insert() 方法执行插入
        collection.insert(data_to_insert)
        # 调用 collection.flush() 确保数据写入 Milvus (对于非 auto-flush 的集合是必要的)
        collection.flush()
        # 打印成功插入的信息和当前集合的总实体数
        print(f"成功插入 {len(new_embeddings)} 个新特征向量（跳过 {skipped_count} 个已存在图像）")
        print(f"集合当前总数：{collection.num_entities}")
    else:
        # 如果没有新图像需要插入，打印提示信息
        print(f"未插入任何新向量，所有 {skipped_count} 个图像已存在")


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 配置区 ---
    # 设置图片所在的目录路径 (请根据实际情况修改为你本地的路径)
    IMAGE_DIRECTORY = "D:\\Code\\heritage\\app_ai\\static\\images"
    # 设置是否强制重新创建集合 (True: 删除旧集合并创建新的, False: 使用现有集合或创建新集合)
    FORCE_RECREATE_COLLECTION = False  # 正常运行时设为 False，需要清空并重建时改为 True
    # --- 配置区结束 ---

    # --- 处理强制重建集合的逻辑 ---
    if FORCE_RECREATE_COLLECTION:
        # 检查集合是否存在
        if utility.has_collection(collection_name):
            # 如果存在，打印提示并删除集合
            print(f"强制删除集合 {collection_name}...")
            utility.drop_collection(collection_name)
        # 打印提示并创建新集合
        print("创建新集合...")
        collection = Collection(name=collection_name, schema=schema)
        # --- 为新创建的集合创建索引并加载 ---
        print(f"为新集合 {collection_name} 创建索引...")
        # 定义索引参数 (同上)
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": 1024
            }
        }
        # 创建索引
        collection.create_index(field_name="embedding",
                                index_params=index_params)
        print(f"已为新集合创建索引 {index_params['index_type']}")
        # 加载新创建并已建立索引的集合到内存，准备进行查询和插入
        print(f"加载新集合 {collection_name}...")
        collection.load()
        print("新集合加载完成")
        # --- 添加结束 ---
    else:
        # --- 如果不是强制重建模式 ---
        # 检查在脚本开头获取的 collection 对象是否有效
        if collection:
            # 加载现有集合到内存
            print(f"加载现有集合 {collection_name}...")
            collection.load()
            print(f"现有集合 {collection_name} 加载完成")
        else:
            # 如果 collection 对象无效 (可能在脚本开头获取失败)，打印错误并退出
            print(f"错误：无法获取集合对象 {collection_name}")
            exit()  # 退出脚本

    # --- 处理指定目录下的所有图片 ---
    image_dir = IMAGE_DIRECTORY  # 使用配置中指定的图片目录
    all_vectors = []  # 用于存储所有提取到的特征向量
    all_image_paths = []  # 用于存储所有处理的图片的完整路径

    # 获取目录下所有符合条件的图片文件列表 (png, jpg, jpeg, webp)
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith((
            '.png', '.jpg', '.jpeg', '.webp'))
    ]  # 添加 .webp 后缀

    # 检查是否找到了图片文件
    if not image_files:
        print(f"在目录 {image_dir} 中未找到任何图片")
    else:
        # 如果找到图片，打印数量并开始提取特征
        print(f"在目录 {image_dir} 中找到 {len(image_files)} 张图片，正在提取特征...")
        # 遍历图片文件列表
        for image_file in image_files:
            # 构建图片的完整路径
            image_path = os.path.join(image_dir, image_file)
            try:
                # 调用 resnet 模块的 extract_features 函数提取特征向量
                query_vector = extract_features(image_path)
                # 将提取到的向量和对应的路径添加到列表中
                all_vectors.append(query_vector)
                all_image_paths.append(image_path)
                # 打印单张图片处理完成的提示
                print(f"已提取图片 {image_file} 的特征")
            except Exception as e:
                # 如果处理单张图片时发生错误，打印错误信息并继续处理下一张
                print(f"处理图片 {image_file} 时出错: {e}")

        # --- 批量插入提取到的特征向量 ---
        # 检查是否成功提取到了任何特征向量
        if all_vectors:
            # 如果提取到了向量，打印提示并调用 insert_vectors 函数进行插入
            print(f"正在向Milvus插入 {len(all_vectors)} 个特征向量...")
            insert_vectors(all_vectors, all_image_paths)
            print("插入完成")  # 打印插入操作完成的提示
        else:
            # 如果未能成功提取任何特征，打印提示
            print("未能成功提取任何特征")
