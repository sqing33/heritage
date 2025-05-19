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

def delete_images_from_milvus_and_fs(collection, image_ids, app_root_path):
    """从Milvus数据库和文件系统中删除选定的图片"""
    if collection is None:
        return {'success': False, 'message': 'Milvus 集合未加载，无法执行删除操作。', 'deleted_count': 0, 'errors': ['Milvus collection not loaded.']}

    deleted_count = 0
    errors = []

    try:
        processed_image_ids = []
        for uid in image_ids:
            try:
                # 验证ID是否为有效整数且大于0
                uid_int = int(uid)
                if uid_int <= 0:
                    errors.append(f"ID '{uid}' 必须是大于0的整数。")
                    continue
                processed_image_ids.append(uid_int)
            except ValueError:
                errors.append(f"ID '{uid}' 不是有效的整数格式。")
        
        if not processed_image_ids:
            if not errors: # 如果 processed_image_ids 为空且没有转换错误，说明原始列表为空
                errors.append("未提供有效的图片ID进行删除。")
            return {'success': False, 'message': '提供的ID列表中没有有效的整数ID。', 'deleted_count': 0, 'errors': errors}

        id_list_str = ", ".join(map(str, processed_image_ids))
        expr = f"id in [{id_list_str}]"
        print(f"查询表达式: {expr}")

        results = collection.query(
            expr=expr,
            output_fields=["id", "image_filename"],
        )
        print(f"查询到 {len(results)} 条记录准备删除")

        if not results:
            errors.append(f"未在Milvus中找到ID为 {id_list_str} 的记录。")
            # 建议用户检查ID是否正确或使用list_images_utils.py列出所有图片ID
            errors.append("建议使用list_images_utils.py脚本列出所有图片ID，确认要删除的ID是否存在。")
            return {'success': True, 'message': '没有与提供的ID匹配的图片可删除。', 'deleted_count': 0, 'errors': errors}

        # 删除Milvus中的记录
        delete_result = collection.delete(expr)
        print(f"Milvus 删除结果: {delete_result}")

        # 删除对应的图片文件
        print(f"开始删除 {len(results)} 个图片文件")
        for item in results:
            try:
                image_filename = item.get('image_filename')
                if not image_filename:
                    errors.append(f"ID {item.get('id')} 的记录缺少 image_filename 字段。")
                    continue
                
                image_path = os.path.join(app_root_path, 'static', 'images', image_filename)
                print(f"尝试删除文件: {image_path}")
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"成功删除文件: {image_filename}")
                    deleted_count += 1
                else:
                    error_msg = f"文件不存在: {image_path}"
                    print(error_msg)
                    errors.append(error_msg)
            except Exception as file_err:
                error_msg = f"删除文件 {item.get('image_filename', '未知文件')} (ID: {item.get('id')}) 失败: {file_err}"
                print(error_msg)
                errors.append(error_msg)
        
        return {'success': True, 'message': f'成功删除 {deleted_count} 个图片', 'deleted_count': deleted_count, 'errors': errors}

    except Exception as e:
        print(f"删除图片过程中发生严重错误: {e}")
        errors.append(f"删除图片失败: {e}")
        return {'success': False, 'message': f'删除图片失败: {e}', 'deleted_count': deleted_count, 'errors': errors}

if __name__ == "__main__":
    print(f"开始测试 Milvus 集合 '{collection_name}' 的删除功能...")
    
    # 1. 连接到 Milvus
    connect_to_milvus()
    
    # 2. 加载集合
    current_collection = load_milvus_collection(collection_name)
    
    if current_collection:
        # 3. 获取应用根路径
        app_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 4. 提示用户输入要删除的ID列表
        print(f"集合 '{collection_name}' 当前包含 {current_collection.num_entities} 条实体。")
        print("请输入要删除的图片ID列表(多个ID用逗号分隔):")
        input_ids = input().strip()
        
        # 5. 处理输入并调用删除函数
        if input_ids:
            image_ids = [id.strip() for id in input_ids.split(',') if id.strip()]
            print(f"准备删除以下ID的图片: {', '.join(image_ids)}")
            
            # 调用删除函数
            result = delete_images_from_milvus_and_fs(current_collection, image_ids, app_root_path)
            
            # 打印删除结果
            print(f"\n删除结果: {result['message']}")
            if result['errors']:
                print("\n错误信息:")
                for error in result['errors']:
                    print(f"- {error}")
        else:
            print("未输入任何ID，删除操作已取消。")
    else:
        print(f"无法加载 Milvus 集合 '{collection_name}'，删除操作中止。")
    
    print("\n删除功能测试结束。")
