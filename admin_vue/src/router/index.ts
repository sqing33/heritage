import { createRouter, createWebHistory } from "vue-router";
import ImageSearch from "@/components/ImageSearch.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "home",
      component: () => import("../components/index.vue"),
    },
    {
      path: "/upload",
      name: "upload",
      component: () => import("../components/ImageUpload.vue"),
    },
    {
      path: "/image",
      name: "image",
      component: () => import("../components/ImageList.vue"),
    },
    {
      path: "/search",
      name: "ImageSearch",
      component: ImageSearch,
    },
  ],
});

export default router;
