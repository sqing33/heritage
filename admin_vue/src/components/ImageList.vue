<template>
  <div>
    <el-affix :offset="80">
      <el-button
        type="primary"
        size="small"
        @click="handleSelectAll"
        style="margin-bottom: 10px; margin-right: 10px"
        >全选</el-button
      >
      <el-button
        type="danger"
        size="small"
        :disabled="selectedImages.length === 0"
        @click="handleBatchDelete"
        style="margin-bottom: 10px"
        >批量删除</el-button
      >
    </el-affix>

    <div class="card-list">
      <div
        v-for="item in imageList"
        :key="item.id"
        class="image-card"
        :class="{ selected: selectedImages.includes(item) }"
        @click="toggleSelect(item)"
      >
        <el-image
          :src="getImageUrl(item.image_filename)"
          style="
            height: 150px;
            width: auto;
            object-fit: contain;
            display: block;
            margin: 0 auto;
          "
          fit="contain"
        />
        <div class="card-info">
          <div class="filename" :title="item.image_filename">
            {{ item.image_filename }}
          </div>
          <div class="id">ID: {{ item.id }}</div>
        </div>
        <div class="card-actions">
          <el-button type="danger" size="small" @click.stop="handleDelete(item)"
            >删除</el-button
          >
        </div>
        <el-checkbox
          class="select-checkbox"
          :model-value="selectedImages.includes(item)"
          @change="(checked) => handleCardSelect(item, checked)"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import axios from "axios";

const imageList = ref<any[]>([]);
const selectedImages = ref<any[]>([]);

const getImageUrl = (filename: string) => {
  // 假设图片都在 /static/images/ 下
  return `http://localhost:5000/static/images/${filename}`;
};

const fetchImages = async () => {
  try {
    const res = await axios.get("http://localhost:5000/api/images");
    if (res.data.success) {
      // id转字符串，防止大整数精度丢失
      imageList.value = (res.data.data || []).map((item: any) => ({
        ...item,
        id: String(item.id),
      }));
    } else {
      ElMessage.error(res.data.message || "获取图片列表失败");
    }
  } catch (e) {
    ElMessage.error("获取图片列表失败");
  }
};

onMounted(fetchImages);

const handleSelectionChange = (val: any[]) => {
  selectedImages.value = val;
};

const handleDelete = (row: any) => {
  ElMessageBox.confirm("确认删除该图片吗?", "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(async () => {
      await deleteImages([row.id]);
    })
    .catch(() => {
      ElMessage.info("已取消删除");
    });
};

const handleBatchDelete = () => {
  if (selectedImages.value.length === 0) return;
  ElMessageBox.confirm(
    `确认删除选中的${selectedImages.value.length}张图片吗?`,
    "提示",
    {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    }
  )
    .then(async () => {
      const ids = selectedImages.value.map((item) => item.id);
      await deleteImages(ids);
    })
    .catch(() => {
      ElMessage.info("已取消删除");
    });
};

const deleteImages = async (ids: number[]) => {
  try {
    const res = await axios.post("http://localhost:5000/api/delete_images", {
      ids,
    });
    if (res.data.success) {
      ElMessage.success(res.data.message || "删除成功");
      fetchImages();
    } else {
      ElMessage.error(res.data.message || "删除失败");
    }
  } catch (e) {
    ElMessage.error("删除失败");
  }
};

// 支持上传后刷新
const emits = defineEmits(["refresh"]);
watch(
  () => emits,
  () => {
    fetchImages();
  }
);

const toggleSelect = (item: any) => {
  const idx = selectedImages.value.indexOf(item);
  if (idx === -1) {
    selectedImages.value.push(item);
  } else {
    selectedImages.value.splice(idx, 1);
  }
};
const handleCardSelect = (item: any, checked: boolean) => {
  if (checked) {
    if (!selectedImages.value.includes(item)) selectedImages.value.push(item);
  } else {
    const idx = selectedImages.value.indexOf(item);
    if (idx !== -1) selectedImages.value.splice(idx, 1);
  }
};

const handleSelectAll = () => {
  if (selectedImages.value.length === imageList.value.length) {
    selectedImages.value = [];
  } else {
    selectedImages.value = [...imageList.value];
  }
};
</script>

<style scoped>
.el-table {
  margin: 20px;
}
.card-list {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
  margin: 20px 0;
}
.image-card {
  width: 220px;
  background: #fff;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  box-shadow: 0 2px 8px #f0f1f2;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px 8px 8px 8px;
  position: relative;
  cursor: pointer;
  transition: box-shadow 0.2s, border-color 0.2s;
}
.image-card.selected {
  border-color: #409eff;
  box-shadow: 0 4px 16px #d0e2ff;
}
.card-info {
  width: 100%;
  margin-top: 10px;
  font-size: 13px;
  color: #333;
  word-break: break-all;
}
.filename {
  font-weight: bold;
  margin-bottom: 4px;
}
.hash {
  color: #888;
  font-size: 12px;
  margin-bottom: 2px;
}
.id {
  color: #bbb;
  font-size: 12px;
}
.card-actions {
  margin-top: 8px;
  width: 100%;
  display: flex;
  justify-content: flex-end;
}
.select-checkbox {
  position: absolute;
  top: 8px;
  right: 8px;
}
</style>
