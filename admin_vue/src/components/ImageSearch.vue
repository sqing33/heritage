<template>
  <div class="search-container">
    <el-upload
      class="upload-demo"
      :action="''"
      :show-file-list="false"
      :before-upload="beforeUpload"
      :http-request="handleSearch"
      :auto-upload="true"
      :multiple="false"
      :drag="true"
    >
      <el-button type="primary">选择图片搜索</el-button>
      <template #tip>
        <div class="el-upload__tip">上传图片后自动搜索相似图片</div>
      </template>
    </el-upload>
    <el-input-number
      v-model="topK"
      :min="1"
      :max="20"
      label="返回数量"
      style="margin: 10px 0"
      size="small"
    />
    <el-divider />
    <div v-if="loading" class="search-loading">
      <el-icon><loading /></el-icon> 正在搜索...
    </div>
    <div v-if="errorMsg" class="search-error">
      <el-alert :title="errorMsg" type="error" show-icon />
    </div>
    <div class="search-result-scroll">
      <div v-if="results.length" class="result-list">
        <div class="result-card" v-for="item in results" :key="item.id">
          <el-image
            :src="item.image_url"
            style="
              height: 120px;
              width: auto;
              object-fit: contain;
              display: block;
              margin: 0 auto;
            "
            fit="contain"
          />
          <div class="result-info">
            <div class="filename" :title="item.filename">
              {{ item.filename }}
            </div>
            <div class="distance">距离: {{ item.distance }}</div>
            <div class="similarity">
              相似度: {{ ((1 - Number(item.distance)) * 100).toFixed(2) }}%
            </div>
          </div>
        </div>
      </div>
      <div v-else-if="!loading && !errorMsg" class="no-result">
        暂无搜索结果
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { ElMessage } from "element-plus";
import type { UploadRequestOptions } from "element-plus";
import axios from "axios";
import { Loading } from "@element-plus/icons-vue";

const results = ref<any[]>([]);
const loading = ref(false);
const errorMsg = ref("");
const topK = ref(5);

const beforeUpload = (file: File) => {
  const isAllowed = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
  ].includes(file.type);
  const isLt16M = file.size / 1024 / 1024 < 16;
  if (!isAllowed) {
    ElMessage.error("只能上传图片文件");
  }
  if (!isLt16M) {
    ElMessage.error("图片大小不能超过16MB");
  }
  return isAllowed && isLt16M;
};

const handleSearch = async (options: UploadRequestOptions) => {
  loading.value = true;
  errorMsg.value = "";
  results.value = [];
  const formData = new FormData();
  formData.append("file", options.file as File);
  formData.append("top_k", String(topK.value));
  try {
    const res = await axios.post("http://localhost:5000/api/search", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    if (res.data.success) {
      results.value = res.data.results || [];
      if (!results.value.length) {
        errorMsg.value = "未找到相似图片";
      }
    } else {
      errorMsg.value = res.data.message || "搜索失败";
    }
  } catch (e: any) {
    errorMsg.value = e?.response?.data?.message || "搜索失败";
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.search-container {
  max-width: 700px;
  margin: 0 auto;
  background: #fff;
  padding: 24px;
  border-radius: 8px;
  box-shadow: 0 2px 10px #f0f1f2;
}
.search-result-scroll {
  max-height: 500px;
  overflow-y: auto;
}
.result-list {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
  margin-top: 20px;
}
.result-card {
  width: 180px;
  background: #fafbfc;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  box-shadow: 0 2px 8px #f0f1f2;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12px 8px 8px 8px;
}
.result-info {
  width: 100%;
  margin-top: 10px;
  font-size: 13px;
  color: #333;
  word-break: break-all;
  text-align: center;
}
.filename {
  font-weight: bold;
  margin-bottom: 4px;
}
.distance {
  color: #888;
  font-size: 12px;
  margin-bottom: 2px;
}
.similarity {
  color: #409eff;
  font-size: 12px;
}
.no-result {
  color: #bbb;
  text-align: center;
  margin-top: 30px;
}
.search-loading {
  color: #409eff;
  text-align: center;
  margin: 30px 0;
  font-size: 18px;
}
.search-error {
  margin: 20px 0;
}
</style>
