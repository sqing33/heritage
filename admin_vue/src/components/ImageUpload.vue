<template>
  <el-upload
    class="upload-demo"
    :action="uploadUrl"
    :on-success="handleSuccess"
    :on-error="handleError"
    :show-file-list="false"
    :before-upload="beforeUpload"
    :headers="headers"
    :data="{}"
    :auto-upload="true"
    :with-credentials="false"
    :multiple="true"
    :http-request="customRequest"
    :drag="true"
    :limit="0"
  >
    <el-button type="primary">点击上传</el-button>
    <template #tip>
      <div class="el-upload__tip">
        可多选，支持png/jpg/jpeg/webp/gif文件，单文件不超过16MB
      </div>
    </template>
  </el-upload>
  <el-table
    v-if="uploadResults.length"
    :data="uploadResults"
    style="width: 100%; margin: 20px 0"
    size="small"
    border
  >
    <el-table-column label="预览" width="120" align="center">
      <template #default="{ row }">
        <el-image
          v-if="row.url"
          :src="row.url"
          style="
            height: 80px;
            width: auto;
            object-fit: contain;
            display: block;
            margin: 0 auto;
          "
          :preview-src-list="[row.url]"
        />
        <span v-else style="color: #bbb">-</span>
      </template>
    </el-table-column>
    <el-table-column
      prop="filename"
      label="文件名"
      width="220"
      align="center"
    />
    <el-table-column prop="status" label="状态" width="100" align="center">
      <template #default="{ row }">
        <el-tag v-if="row.status === 'success'" type="success">成功</el-tag>
        <el-tag v-else-if="row.status === 'exist'" type="warning"
          >已存在</el-tag
        >
        <el-tag v-else-if="row.status === 'error'" type="danger">失败</el-tag>
        <el-tag v-else type="info">上传中</el-tag>
      </template>
    </el-table-column>
    <el-table-column prop="message" label="消息" align="center" />
  </el-table>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { ElMessage } from "element-plus";
import type { UploadRequestOptions } from "element-plus";
import axios from "axios";

const uploadUrl = "http://localhost:5000/insert_image";
const headers = {};

const emit = defineEmits(["uploaded"]);

interface UploadResult {
  filename: string;
  status: "success" | "error" | "uploading" | "exist";
  message?: string;
  url?: string;
}

const uploadResults = ref<UploadResult[]>([]);

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

// 自定义上传，支持多文件并发，逐个显示状态和图片预览
const customRequest = async (options: UploadRequestOptions) => {
  const file = options.file as File;
  // 生成本地预览URL
  const localUrl = URL.createObjectURL(file);
  const row: UploadResult = {
    filename: file.name,
    status: "uploading",
    message: "",
    url: localUrl,
  };
  uploadResults.value.push(row);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await axios.post(uploadUrl, formData, {
      headers: {
        ...headers,
        "Content-Type": "multipart/form-data",
      },
    });
    const idx = uploadResults.value.indexOf(row);
    if (res.data && res.data.success) {
      const newRow = {
        ...row,
        status: "success",
        message: res.data.message || "上传成功",
        // 如果后端返回了图片路径，可以替换为后端路径
        url: row.url,
      };
      if (idx !== -1) uploadResults.value.splice(idx, 1, newRow);
      emit("uploaded");
    } else if (
      res.data &&
      res.data.message &&
      res.data.message.includes("已存在")
    ) {
      const newRow = {
        ...row,
        status: "exist",
        message: res.data.message,
        url: row.url,
      };
      if (idx !== -1) uploadResults.value.splice(idx, 1, newRow);
    } else {
      const newRow = {
        ...row,
        status: "error",
        message: res.data.message || "上传失败",
        url: row.url,
      };
      if (idx !== -1) uploadResults.value.splice(idx, 1, newRow);
    }
  } catch (e: any) {
    const idx = uploadResults.value.indexOf(row);
    const newRow = {
      ...row,
      status: "error",
      message: e?.response?.data?.message || "上传失败",
      url: row.url,
    };
    if (idx !== -1) uploadResults.value.splice(idx, 1, newRow);
  }
};

const handleSuccess = () => {
  // 不使用 el-upload 默认的 handleSuccess
};

const handleError = () => {
  // 不使用 el-upload 默认的 handleError
};
</script>

<style scoped>
.upload-demo {
  margin: 20px;
}

.el-table th,
.el-table td {
  text-align: center;
  vertical-align: middle;
}
.upload-result {
  margin: 0 20px;
}
</style>
