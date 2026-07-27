import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  /* config options here */
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
  // Disable Turbopack for production builds to avoid module resolution issues
  experimental: {
    turbo: undefined,
  },
};

export default nextConfig;
