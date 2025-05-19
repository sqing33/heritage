<template>
  <el-header>
    <div class="header-content">
      <h2>图片管理系统</h2>
      <el-dropdown>
        <el-button type="primary">
          管理员
          <el-icon class="el-icon--right"><arrow-down /></el-icon>
        </el-button>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item>个人信息</el-dropdown-item>
            <el-dropdown-item>退出登录</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>
  </el-header>
  <el-container>
    <el-aside width="200px">
      <el-menu default-active="/" class="el-menu-vertical" router>
        <el-menu-item index="/">
          <el-icon><Orange /></el-icon>
          <span>首页</span>
        </el-menu-item>
        <el-menu-item index="/upload">
          <el-icon><UploadFilled /></el-icon>
          <span>图片上传</span>
        </el-menu-item>
        <el-menu-item index="/image">
          <el-icon><PictureFilled /></el-icon>
          <span>图片管理</span>
        </el-menu-item>
        <el-menu-item index="/search">
          <el-icon><Search /></el-icon>
          <span>图片搜索</span>
        </el-menu-item>
        <el-menu-item index="/settings">
          <el-icon><setting /></el-icon>
          <span>系统设置</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-main class="main-scroll">
      <el-scrollbar style="height: 100%">
        <router-view />
      </el-scrollbar>
    </el-main>
  </el-container>
</template>

<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";
import ImageUpload from "@/components/ImageUpload.vue";
import ImageList from "@/components/ImageList.vue";
import {
  ArrowDown,
  Picture,
  Setting,
  UploadFilled,
  Search,
  Orange,
  PictureFilled,
} from "@element-plus/icons-vue";
import { ElScrollbar } from "element-plus";
</script>

<style>
html,
body {
  height: 100%;
  margin: 0;
  padding: 0;
  overflow: hidden; /* 禁止页面滚动条 */
}
</style>

<style scoped>
.layout-container {
  min-height: 100vh;
  /* 防止外层滚动 */
  overflow: hidden;
}

.el-header {
  background-color: #fff;
  border-bottom: 1px solid #dcdfe6;
  padding: 0 20px;
}

.header-content {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.el-aside {
  background-color: #fff;
}

.el-menu-vertical {
  border-right: none;
}

.el-main.main-scroll {
  background-color: #f0f2f5;
  padding: 20px;
  height: calc(100vh - 60px);
  box-sizing: border-box;
  overflow: hidden;
}
</style>
