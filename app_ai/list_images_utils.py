import os
from pymilvus import connections, Collection, utility

# --- Milvus 连接配置 ---
def connect_to_milvus(host='192.168.1.100', port='19530', alias='default'):
    """连接到 Milvus 服务器"""
    if not connections.has_connection(alias):
        print(f"正在连接到 Milvus 服务器 {host}:{port}...")
        connections.connect(alias=alias, host=host, port=port)
        print("成功连接到 Milvus。")
    else:
        print("已存在 Milvus 连接。")

# --- Milvus 集合配置 ---
collection_name = "intangible_cultural_heritage_images"  # 与 app_flask.py 和其他脚本保持一致

def load_milvus_collection(collection_name_to_load):
    """加载指定的 Milvus 集合"""
    if utility.has_collection(collection_name_to_load):
        collection = Collection(name=collection_name_to_load)
        try:
            collection.load()
            print(f"成功加载集合 {collection_name_to_load}。")
            return collection
        except Exception as load_err:
            print(f"加载集合 {collection_name_to_load} 失败: {load_err}")
            return None
    else:
        print(f"错误：集合 {collection_name_to_load} 不存在。")
        return None

def list_all_images_from_milvus(collection):
    """从 Milvus 数据库中列出所有图片及其 ID 和文件名"""
    if collection is None:
        print("Milvus 集合未加载，无法执行列出操作。")
        return []

    images_list = []
    try:
        # 查询所有实体，获取 id 和 image_filename 字段
        # Milvus 的 query 接口支持迭代器，可以处理大量数据
        # 为了简单起见，这里我们一次性获取所有数据，但对于非常大的集合，可能需要分批处理
        # 使用 expr="id > 0" 或类似的表达式来获取所有记录，或者不指定 expr 来获取所有记录
        # output_fields 指定了我们想要检索的字段
        # consistency_level="Strong" 确保我们读取到最新的数据
        print(f"正在从集合 '{collection.name}' 中查询所有图片信息...")
        results = collection.query(
            expr="",  # 空表达式通常意味着查询所有记录，具体行为可能依赖于 Milvus 版本和配置
            output_fields=["id", "image_filename"],
            consistency_level="Strong", # 确保读取到最新的数据
            limit=1000  # 添加 limit 参数以避免错误
        )
        
        print(f"查询到 {len(results)} 条记录。")

        for item in results:
            images_list.append({
                "id": item.get('id'),
                "image_filename": item.get('image_filename')
            })
        
        return images_list

    except Exception as e:
        print(f"从 Milvus 列出图片过程中发生错误: {e}")
        return []

if __name__ == "__main__":
    print("开始列出 Milvus 数据库中的图片...")
    
    # 1. 连接到 Milvus
    connect_to_milvus()
    
    # 2. 加载集合
    current_collection = load_milvus_collection(collection_name)
    
    if current_collection:
        # 3. 列出所有图片
        all_images = list_all_images_from_milvus(current_collection)
        
        # 4. 打印结果
        if all_images:
            print("\n--- 数据库中的图片列表 ---")
            for image_info in all_images:
                print(f"ID: {image_info['id']}, 文件名: {image_info['image_filename']}")
            print(f"\n总共找到 {len(all_images)} 张图片。")
        else:
            print("数据库中没有找到图片，或者在列出过程中发生错误。")
    else:
        print("无法加载 Milvus 集合，列出操作中止。")

    print("\n图片列出功能测试结束。")
