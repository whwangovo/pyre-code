'use client';

import katex from 'katex';
import 'katex/dist/katex.min.css';
import { memo, type ReactNode } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { PythonCode } from '@/lib/pythonHighlight';

interface MarkdownContentProps {
  content: string;
}

// Models often write \( \) and \[ \]; remark-math only knows $ and $$. Code spans and fences
// are skipped so Python like `x[\[0\]]` or regexes stay untouched.
function normalizeMath(content: string): string {
  return content
    .replace(/\r\n/g, '\n')
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g)
    .map((part, i) => {
      if (i % 2 === 1) return part;
      return part
        .replace(/\\\[([\s\S]+?)\\\]/g, (_, tex: string) => `\n$$\n${tex.trim()}\n$$\n`)
        .replace(/\\\(([\s\S]+?)\\\)/g, (_, tex: string) => `$${tex.trim()}$`);
    })
    .join('');
}

function MathFormula({ tex, display }: { tex: string; display: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false });
  return display
    ? <div className="overflow-x-auto overflow-y-hidden py-1 [&_.katex-display]:my-0" dangerouslySetInnerHTML={{ __html: html }} />
    : <span dangerouslySetInnerHTML={{ __html: html }} />;
}

function CodeBlock({ language, content }: { language: string; content: string }) {
  const isPython = language === '' || language === 'python' || language === 'py';
  return (
    <div className="overflow-hidden rounded-xl" style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}>
      {language && (
        <div className="px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-text-3" style={{ borderBottom: '1px solid var(--line)' }}>
          {language}
        </div>
      )}
      <pre className="overflow-x-auto px-4 py-3 text-xs leading-relaxed">
        {isPython ? <PythonCode code={content} /> : <code className="font-mono text-text">{content}</code>}
      </pre>
    </div>
  );
}

function textOf(children: ReactNode): string {
  if (typeof children === 'string') return children;
  if (Array.isArray(children)) return children.map(textOf).join('');
  return '';
}

const components: Components = {
  h1: ({ children }) => <div className="text-lg font-semibold text-text">{children}</div>,
  h2: ({ children }) => <div className="text-base font-semibold text-text">{children}</div>,
  h3: ({ children }) => <div className="text-sm font-semibold text-text">{children}</div>,
  h4: ({ children }) => <div className="text-sm font-semibold text-text">{children}</div>,
  p: ({ children }) => <p className="text-sm leading-relaxed text-text-2">{children}</p>,
  ul: ({ children }) => <ul className="space-y-1.5 pl-5 text-sm leading-relaxed text-text-2 list-disc">{children}</ul>,
  ol: ({ children, start }) => <ol start={start} className="space-y-1.5 pl-5 text-sm leading-relaxed text-text-2 list-decimal">{children}</ol>,
  li: ({ children }) => <li className="pl-0.5 [&>ul]:mt-1.5 [&>ol]:mt-1.5">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-text">{children}</strong>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent underline underline-offset-2">{children}</a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="pl-3 text-text-3 space-y-2" style={{ borderLeft: '3px solid var(--line)' }}>{children}</blockquote>
  ),
  hr: () => <hr style={{ borderColor: 'var(--line)' }} />,
  table: ({ children }) => (
    <div className="overflow-x-auto rounded-lg" style={{ border: '1px solid var(--line)' }}>
      <table className="w-full border-collapse text-sm text-text-2">{children}</table>
    </div>
  ),
  th: ({ children, style }) => (
    <th className="px-3 py-2 text-left font-semibold text-text" style={{ ...style, background: 'var(--bg-sunken)', borderBottom: '1px solid var(--line)' }}>{children}</th>
  ),
  td: ({ children, style }) => (
    <td className="px-3 py-2 align-top" style={{ ...style, borderTop: '1px solid var(--line)' }}>{children}</td>
  ),
  // Fenced code and $$ display math both arrive as <pre><code>; render them without the <pre>
  pre: ({ children }) => <>{children}</>,
  code: ({ className, children }) => {
    const classes = className?.split(' ') ?? [];
    const text = textOf(children);
    if (classes.includes('math-display')) return <MathFormula tex={text} display />;
    if (classes.includes('math-inline')) return <MathFormula tex={text} display={false} />;
    const language = classes.find((c) => c.startsWith('language-'))?.slice('language-'.length);
    // Fenced blocks end with a newline (or carry a language); inline code spans never do
    if (language !== undefined || text.endsWith('\n')) {
      return <CodeBlock language={language?.toLowerCase() ?? ''} content={text.replace(/\n$/, '')} />;
    }
    return (
      <code
        className="rounded px-1.5 py-0.5 font-mono text-[0.85em] text-text"
        style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
      >
        {children}
      </code>
    );
  },
};

// Memoized: parsing Markdown + KaTeX is expensive, and callers re-render often while
// streaming AI replies. `content` is a string, so unchanged messages always skip the parse.
export const MarkdownContent = memo(function MarkdownContent({ content }: MarkdownContentProps) {
  return (
    <div className="space-y-3 break-words">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]} components={components}>
        {normalizeMath(content)}
      </ReactMarkdown>
    </div>
  );
});
