import type { Metadata } from "next";
import "./globals.css";
import { LocaleProvider } from "@/context/LocaleContext";
import { ThemeProvider } from "@/context/ThemeContext";
import { DesignProvider } from "@/context/DesignContext";

export const metadata: Metadata = {
  title: "Pyre Code",
  description: "68 hands-on AI systems challenges — implement the internals of attention, RLHF, diffusion, and distributed training",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('pyre-theme');if(t==='dark')document.documentElement.setAttribute('data-theme','dark');var d=localStorage.getItem('pyre-design');if(d==='classic')document.documentElement.setAttribute('data-design','classic')}catch(e){}})()`,
          }}
        />
      </head>
      <body className="min-h-screen">
        <DesignProvider>
          <ThemeProvider>
            <LocaleProvider>{children}</LocaleProvider>
          </ThemeProvider>
        </DesignProvider>
      </body>
    </html>
  );
}
