import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "@/components/layout/theme-provider";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "GLM Platform | Multimodal AI Training & Deployment",
  description: "Complete multimodal conversational AI platform for training, fine-tuning, and deploying GLM models with vision, speech, emotion, and video generation capabilities.",
  keywords: ["GLM", "AI", "Multimodal", "Training", "Fine-tuning", "LLM", "Vision", "Speech", "Video Generation"],
  authors: [{ name: "GLM Platform Team" }],
  openGraph: {
    title: "GLM Platform - Multimodal AI Training & Deployment",
    description: "Train and deploy state-of-the-art multimodal AI models",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <div className="min-h-screen flex flex-col">
            {children}
          </div>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
