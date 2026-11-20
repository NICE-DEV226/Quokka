import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  base: './',
  build: { outDir: 'dist', assetsDir: 'assets', target: 'es2020' },
  server: { proxy: { '/v1': 'http://127.0.0.1:11435', '/health': 'http://127.0.0.1:11435' } }
});
