import { createRouter, createWebHistory } from 'vue-router'
import { auth } from '../stores/auth'

const routes = [
  { path: '/login', name: 'login', component: () => import('../views/LoginView.vue'), meta: { public: true } },
  {
    path: '/',
    component: () => import('../layout/MainLayout.vue'),
    children: [
      { path: '', redirect: '/home' },
      { path: 'home', name: 'home', component: () => import('../views/HomeView.vue'), meta: { title: '首页推荐' } },
      { path: 'records', name: 'records', component: () => import('../views/RecordsView.vue'), meta: { title: '我的追番' } },
      { path: 'new', name: 'new', component: () => import('../views/NewAnimeView.vue'), meta: { title: '新番专区' } },
      { path: 'interest', name: 'interest', component: () => import('../views/InterestView.vue'), meta: { title: '兴趣分析' } },
      { path: 'chat', name: 'chat', component: () => import('../views/ChatView.vue'), meta: { title: '追番助手' } },
      { path: 'admin/users', name: 'admin-users', component: () => import('../views/admin/UsersView.vue'), meta: { title: '用户管理', admin: true } },
      { path: 'admin/animes', name: 'admin-animes', component: () => import('../views/admin/AnimesView.vue'), meta: { title: '番库管理', admin: true } },
      { path: 'admin/monitor', name: 'admin-monitor', component: () => import('../views/admin/MonitorView.vue'), meta: { title: '效果监控', admin: true } },
    ],
  },
  { path: '/:pathMatch(.*)*', redirect: '/home' },
]

const router = createRouter({ history: createWebHistory(), routes })

router.beforeEach((to) => {
  if (!to.meta.public && !auth.isLogin) return { name: 'login', query: { redirect: to.fullPath } }
  if (to.meta.admin && !auth.isAdmin) return { name: 'home' }
})

export default router
