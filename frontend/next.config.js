/** @type {import('next').NextConfig} */
const API_PROXY_TARGET = (process.env.API_PROXY_TARGET || "http://127.0.0.1:8001").replace(
  /\/$/,
  "",
);

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api-proxy/:path*",
        destination: `${API_PROXY_TARGET}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
