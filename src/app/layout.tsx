import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "GLM Platform - Free Multimodal AI Assistant",
  description: "GLM Platform - A free, open-source multimodal AI platform powered by Z.ai SDK. Chat, generate images, and more without API keys!",
  keywords: ["GLM", "AI", "ChatGPT alternative", "free AI", "Z.ai", "multimodal AI", "image generation", "Next.js"],
  authors: [{ name: "GLM Platform Team" }],
  icons: {
    icon: "/favicon.ico",
  },
  openGraph: {
    title: "GLM Platform - Free Multimodal AI Assistant",
    description: "Chat, generate images, and more with our free AI platform powered by Z.ai SDK",
    url: "https://conversationalglm2.onrender.com",
    siteName: "GLM Platform",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Prevent flash of wrong theme
              (function() {
                try {
                  const stored = JSON.parse(localStorage.getItem('glm-platform-storage') || '{}');
                  const theme = stored?.state?.settings?.theme || 'dark';
                  document.documentElement.classList.toggle('dark', theme === 'dark');
                } catch(e) {
                  document.documentElement.classList.add('dark');
                }
              })();
            `,
          }}
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground min-h-screen`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
