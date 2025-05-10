import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from pymilvus import connections, Collection, utility  # 导入 Milvus 相关库
from resnet import extract_features  # 导入特征提取函数
# 注意：直接从 search_images 导入 search_similar_vectors 函数
# search_images.py 中的顶层代码（连接和加载）会在导入时执行一次
from search_images import search_similar_vectors, collection_name

# --- Flask 应用配置 ---
# Define UPLOAD_FOLDER relative to this file's directory (app_ai) to get an absolute path
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}  # 允许上传的文件类型

app = Flask(__name__, template_folder='templates', static_folder='static'
            )  # static_folder relative to app location (app_ai/static)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Store the absolute path
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为 16MB
app.secret_key = 'super secret key'  # 用于 flash 消息，生产环境应使用更安全的密钥

# 确保上传目录存在 using the absolute path
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Milvus 连接和集合加载 (在应用启动时执行) ---
collection = None  # 初始化 collection 变量
try:
    # 检查是否已存在连接，如果不存在则创建
    if not connections.has_connection("default"):
        connections.connect(alias="default",
                            host='192.168.1.100',
                            port='19530')
        print("成功连接到 Milvus。")
    else:
        print("已存在 Milvus 连接。")

    # 检查集合是否存在
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        print(f"集合 {collection_name} 存在。")

        # 尝试加载集合
        try:
            collection.load()
            print(f"成功加载集合 {collection_name}。")
        except Exception as load_err:
            print(f"加载集合 {collection_name} 失败: {load_err}")
            collection = None  # 确保在加载失败时将 collection 设置为 None
            flash(f"加载 Milvus 集合失败: {load_err}")  # 通过 flash 显示错误

    else:
        print(f"错误：集合 {collection_name} 不存在。请先运行 insert_images.py。")
        flash(f"错误：集合 {collection_name} 不存在。请先运行 insert_images.py创建集合。")
        collection = None  # 确保在集合不存在时将 collection 设置为 None

except Exception as e:
    print(f"连接到 Milvus 或加载集合时出错: {e}")
    flash(f"连接到 Milvus 或加载集合时出错: {e}")
    collection = None  # 确保在出现任何错误时将 collection 设置为 None


# --- 辅助函数 ---
def allowed_file(filename):
    """检查文件扩展名是否在允许范围内"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Flask 路由 ---
@app.route('/')
def upload_form():
    """显示文件上传表单"""
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """处理图片上传、特征提取和相似度搜索"""
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
        # Use the absolute path from config for saving and processing
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            # 将文件保存到静态文件夹 using the absolute path
            file.save(filepath)
            flash(f'文件 {filename} 上传成功，正在处理...')

            # 1. 提取特征
            print(f"正在提取上传图片 {filepath} 的特征...")
            query_vector = extract_features(filepath)

            # 2. 执行搜索
            print(f"正在搜索相似图像...")
            # 从表单获取 top_k 值，如果未提供或无效，则默认为 5
            try:
                top_k = int(request.form.get('top_k', 5))
                if top_k <= 0 or top_k > 50:  # 限制 top_k 范围
                    top_k = 5
            except ValueError:
                top_k = 5

            similar_results = search_similar_vectors(query_vector, top_k=top_k)

            # 3. 准备结果以供显示
            # 可以添加逻辑来生成相似图片的 URL 或路径，以便在模板中显示它们
            # 例如，如果你的 Milvus 数据包含相对路径或 ID 可以映射到文件
            results_for_template = []
            for res in similar_results:
                # 假设 filename 字段存储的是可以直接访问的文件名或相对路径
                # 你可能需要根据实际情况调整这里的路径构造
                image_url = url_for('static',
                                    filename=f'images/{res["filename"]}'
                                    )  # 假设原图在 static/images 下
                results_for_template.append({
                    'id': res['id'],
                    'distance': res['distance'],
                    'filename': res['filename'],
                    'image_url': image_url  # 添加图片 URL
                })

            # 渲染结果页面
            return render_template(
                'results.html',
                results=results_for_template,
                query_filename=filename,
                query_image_url=url_for(
                    'static', filename=f'uploads/{filename}'))  # 传递上传图片的 URL

        except FileNotFoundError:
            flash(f"错误：处理文件时未找到：{filepath}")
            return redirect(url_for('upload_form'))
        except Exception as e:
            flash(f'处理文件或执行搜索时出错: {e}')
            print(f"错误详情: {e}")  # 在服务器日志中打印详细错误
            return redirect(url_for('upload_form'))
        finally:
            # 可选：处理完后删除上传的文件以节省空间
            # if os.path.exists(filepath):
            #     os.remove(filepath)
            pass

    else:
        flash('不允许的文件类型')
        return redirect(request.url)


# Flask's default static file handling will serve files from the 'static_folder' ('app_ai/static')
# at the '/static' URL prefix. This includes subdirectories like 'images' and 'uploads'.
# The custom route for '/static/uploads/' is removed as it should be handled by the default mechanism.
# The custom route for '/static/images/' was already removed.

# Flask's default static file handling will serve files from the 'static_folder' ('app_ai/static')
# at the '/static' URL prefix. This includes subdirectories like 'images'.
# The custom route for '/static/images/' is removed as it's redundant and was pointing
# to the wrong directory ('app_ai/images' instead of 'app_ai/static/images').

if __name__ == '__main__':
    # 使用 waitress 或 gunicorn 等生产级 WSGI 服务器运行应用
    # 例如: waitress-serve --host 0.0.0.0 --port 5000 app_ai.app_flask:app
    app.run(debug=True, host='0.0.0.0', port=5000)  # 仅用于开发
