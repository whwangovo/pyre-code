import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
      },
      colors: {
        bg: {
          DEFAULT: 'var(--bg)',
          elev: 'var(--bg-elev)',
          sunken: 'var(--bg-sunken)',
          code: 'var(--bg-code)',
        },
        line: {
          DEFAULT: 'var(--line)',
          strong: 'var(--line-strong)',
        },
        text: {
          DEFAULT: 'var(--text)',
          '2': 'var(--text-2)',
          '3': 'var(--text-3)',
        },
        accent: {
          DEFAULT: 'var(--accent)',
          ink: 'var(--accent-ink)',
          wash: 'var(--accent-wash)',
          line: 'var(--accent-line)',
          hover: '#0062cc',
        },
        easy: 'var(--easy)',
        medium: 'var(--medium)',
        hard: 'var(--hard)',
        // classic design tokens
        surface: { DEFAULT: '#ffffff', secondary: '#fafafa' },
        border: '#e5e5e5',
        'text-primary': '#1d1d1f',
        'text-secondary': '#6e6e73',
        'text-tertiary': '#aeaeb2',
        solved: '#30b0c7',
      },
      boxShadow: {
        soft: '0 1px 3px rgba(0,0,0,0.06)',
        'soft-lg': '0 4px 12px rgba(0,0,0,0.08)',
      },
      borderColor: {
        DEFAULT: 'var(--line)',
      },
      borderRadius: {
        pill: '9999px',
      },
      animation: {
        ticker: 'ticker 60s linear infinite',
      },
      keyframes: {
        ticker: {
          from: { transform: 'translateX(0)' },
          to: { transform: 'translateX(-50%)' },
        },
      },
    },
  },
  plugins: [],
};
export default config;
