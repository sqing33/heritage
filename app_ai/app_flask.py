import os
import shutil
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, current_app
from werkzeug.utils import secure_filename
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from resnet import extract_features
from search_images import search_similar_vectors, collection_name
from insert_images import insert_vectors
from delete_utils import delete_images_from_milvus_and_fs
from flask_cors import CORS

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['APP_ROOT'] = APP_ROOT
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

collection = None
try:
    if not connections.has_connection("default"):
        connections.connect(alias="default",
                            host='192.168.1.100',
                            port='19530')
        print("成功连接到 Milvus。")
    else:
        print("已存在 Milvus 连接。")

    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        print(f"集合 {collection_name} 存在。")

        try:
            collection.load()
            print(f"成功加载集合 {collection_name}。")
        except Exception as load_err:
            print(f"加载集合 {collection_name} 失败: {load_err}")
            collection = None
            flash(f"加载 Milvus 集合失败: {load_err}")

    else:
        print(f"集合 {collection_name} 不存在，开始自动创建...")

        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": 1024
            }
        }
        collection.create_index(field_name="embedding",
                                index_params=index_params)
        collection.load()  # Load the collection after creating index
        print(f"成功创建集合 {collection_name} 并建立索引")
        flash(f"已自动创建新集合 {collection_name}，并完成初始化")
        print(f"集合初始化完成，当前实体数量：{collection.num_entities}")

except Exception as e:
    print(f"集合初始化失败: {e}")
    flash(f"集合初始化失败: {str(e)}。请检查 Milvus 服务状态")
    collection = None
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"已清理无效集合 {collection_name}")


def allowed_file(filename):
    """检查文件扩展名是否在允许范围内"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


# --- Flask 路由 ---
@app.route('/')
def upload_form():
    """显示文件上传表单"""
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """处理图片上传、特征提取和相似度搜索"""
    upload_folder = app.config['UPLOAD_FOLDER']

    if collection is None:
        flash('Milvus 集合未加载，无法执行搜索。请检查服务器状态和集合是否存在。')
        return redirect(url_for('upload_form'))

    if 'file' not in request.files:
        flash('请求中没有文件部分')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('未选择文件')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        try:
            file.save(filepath)
            flash(f'文件 {filename} 上传成功，正在处理...')

            print(f"正在提取上传图片 {filepath} 的特征...")
            query_vector = extract_features(filepath)

            print(f"正在搜索相似图像...")
            try:
                top_k = int(request.form.get('top_k', 5))
                if top_k <= 0 or top_k > 50:
                    top_k = 5
            except ValueError:
                top_k = 5

            similar_results = search_similar_vectors(collection,
                                                     query_vector,
                                                     top_k=top_k)

            results_for_template = []
            for res in similar_results:
                image_url = url_for('static',
                                    filename=f'images/{res["filename"]}')
                results_for_template.append({
                    'id': res['id'],
                    'distance': res['distance'],
                    'filename': res['filename'],
                    'image_url': image_url
                })

            return render_template('results.html',
                                   results=results_for_template,
                                   query_filename=filename,
                                   query_image_url=url_for(
                                       'static',
                                       filename=f'uploads/{filename}'))

        except FileNotFoundError:
            flash(f"错误：处理文件时未找到：{filepath}")
            return redirect(url_for('upload_form'))
        except Exception as e:
            flash(f'处理文件或执行搜索时出错: {e}')
            print(f"错误详情: {e}")
            return redirect(url_for('upload_form'))
        finally:
            pass

    else:
        flash('不允许的文件类型')
        return redirect(request.url)


@app.route('/api/images', methods=['GET'])
def get_all_images():
    """获取Milvus中存储的所有图片数据"""
    if collection is None:
        return jsonify({'success': False, 'message': 'Milvus 集合未加载。'}), 500

    try:
        results = collection.query(
            expr="",
            output_fields=["id", "image_filename", "image_hash"],
            consistency_level="Strong",  # 确保读取到最新的数据
            limit=1000)
        # 将id转为字符串，避免前端精度丢失
        for item in results:
            if "id" in item:
                item["id"] = str(item["id"])
        return jsonify({'success': True, 'data': results}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取图片数据失败: {e}'}), 500


@app.route('/api/delete_images', methods=['POST'])
def delete_images_route():
    """处理从Milvus数据库和文件系统中删除选定图片的请求"""
    app_root = app.config['APP_ROOT']

    if collection is None:
        return jsonify({
            'success': False,
            'message': 'Milvus 集合未加载，无法执行删除操作。'
        }), 500

    data = request.get_json()
    if not data or 'ids' not in data or not isinstance(
            data['ids'], list) or not data['ids']:
        return jsonify({'success': False, 'message': '未提供有效的图片ID列表'}), 400

    image_ids = data['ids']

    result = delete_images_from_milvus_and_fs(collection, image_ids, app_root)

    if result['success']:
        return jsonify({
            'success': True,
            'message': result['message'],
            'deleted_count': result.get('deleted_count', 0),
            'errors': result.get('errors', [])
        }), 200
    else:
        return jsonify({
            'success': False,
            'message': result['message'],
            'deleted_count': result.get('deleted_count', 0),
            'errors': result.get('errors', [])
        }), 500


@app.route('/insert_image', methods=['POST'])
def insert_image_route():
    """处理图片上传、特征提取和插入到 Milvus"""
    upload_folder = app.config['UPLOAD_FOLDER']
    app_root = app.config['APP_ROOT']

    if collection is None:
        return jsonify({
            'success': False,
            'message': 'Milvus 集合未加载，无法执行插入。请检查服务器状态和集合是否存在。'
        }), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '请求中没有文件部分'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        try:
            file.save(filepath)
            print(f'文件 {filename} 上传成功，正在处理并插入...')

            print(f"正在提取上传图片 {filepath} 的特征...")
            query_vector = extract_features(filepath)

            from insert_images import calculate_image_hash, insert_vectors
            image_hash = calculate_image_hash(filepath)
            # 调用 insert_vectors 并获取返回值
            insert_result = insert_vectors([query_vector.tolist()], [filepath])

            target_dir = os.path.join(app_root, 'static', 'images')
            os.makedirs(target_dir, exist_ok=True)

            target_image_path = os.path.join(target_dir, filename)
            shutil.copy(filepath, target_image_path)
            print(f'图片已复制到 {target_image_path}')

            # 根据插入结果返回不同的消息
            if insert_result["inserted"]:
                return jsonify({
                    'success': True,
                    'message': f'图片 {filename} 特征已提取并插入到 Milvus。'
                }), 200
            elif insert_result["skipped"]:
                return jsonify({
                    'success': False,
                    'message': f'图片 {filename} 已存在，未重复插入。'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': f'图片 {filename} 未能插入，原因未知。'
                }), 200

        except FileNotFoundError:
            return jsonify({
                'success': False,
                'message': f'错误：处理文件时未找到：{filepath}'
            }), 500
        except Exception as e:
            print(f"错误详情: {e}")
            return jsonify({
                'success': False,
                'message': f'处理文件或执行插入时出错: {e}'
            }), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    else:
        return jsonify({'success': False, 'message': '不允许的文件类型'}), 400


@app.route('/api/search', methods=['POST'])
def api_search_similar_images():
    """
    接收图片文件和top_k，返回相似图片列表
    """
    if collection is None:
        return jsonify({'success': False, 'message': 'Milvus 集合未加载。'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '请求中没有文件部分'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400

    try:
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        try:
            top_k = int(request.form.get('top_k', 5))
            if top_k <= 0 or top_k > 50:
                top_k = 5
        except Exception:
            top_k = 5

        # 提取特征并搜索
        query_vector = extract_features(temp_path)
        # 注意：search_similar_vectors 只传 query_vector 和 top_k
        results = search_similar_vectors(query_vector, top_k=top_k)

        # 构造图片URL
        for res in results:
            res['image_url'] = url_for('static',
                                       filename=f'images/{res["filename"]}',
                                       _external=True)

        return jsonify({'success': True, 'results': results}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': f'搜索失败: {e}'}), 500
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
