import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const rawProxyPrefix = env.VITE_PROXY_PREFIX || '/'
  const withLeadingSlash = rawProxyPrefix.startsWith('/') ? rawProxyPrefix : `/${rawProxyPrefix}`
  const proxyPrefix = withLeadingSlash.endsWith('/') ? withLeadingSlash : `${withLeadingSlash}/`
  const proxyPrefixRoot = proxyPrefix === '/' ? '' : proxyPrefix.slice(0, -1)
  const publicOrigin = env.VITE_PUBLIC_ORIGIN?.replace(/\/$/, '')
  const serverOrigin = publicOrigin
    // ? `${publicOrigin}${proxyPrefix === '/' ? '' : proxyPrefix.slice(0, -1)}`
    // : undefined
  const backendTarget = env.VITE_BACKEND_TARGET || 'http://127.0.0.1:8001'
  const rawApiBase = env.VITE_API_BASE || `${proxyPrefixRoot}/managedb`
  const apiBase = rawApiBase.startsWith('/') ? rawApiBase.replace(/\/$/, '') : `/${rawApiBase.replace(/\/$/, '')}`

  const proxy = {
    '/api': {
      target: backendTarget,
      changeOrigin: true,
    },
    [`${apiBase}/api`]: {
      target: backendTarget,
      changeOrigin: true,
    },
  }

  return {
    plugins: [react()],
    // Prefix used when app is served behind a reverse proxy path, for example /kooplex/.
    // Set via env: VITE_PROXY_PREFIX=/kooplex/ (or VITE_BASE_URL=/kooplex/).
    // If dev is exposed through an external proxy URL, also set VITE_PUBLIC_ORIGIN.
    base: proxyPrefix,

    // ... other config
    server: {
      host: '0.0.0.0',
      port: 9000,
      allowedHosts: ['k8plex-veo.vo.elte.hu'],
      origin: serverOrigin,
      proxy,
    }
  }
})
