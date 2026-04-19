'use client';

import type { ReactNode } from 'react';
import { PythonCode } from '@/lib/pythonHighlight';

interface MarkdownContentProps {
  content: string;
}

type Block =
  | { type: 'heading'; level: 1 | 2 | 3; content: string }
  | { type: 'list'; items: string[] }
  | { type: 'paragraph'; content: string }
  | { type: 'code'; language: string; content: string };

function parseBlocks(content: string): Block[] {
  const normalized = content.replace(/\r\n/g, '\n');
  const lines = normalized.split('\n');
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) { i += 1; continue; }

    if (trimmed.startsWith('```')) {
      const language = trimmed.slice(3).trim().toLowerCase();
      const codeLines: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length && lines[i].trim().startsWith('```')) i += 1;
      blocks.push({ type: 'code', language, content: codeLines.join('\n') });
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      blocks.push({ type: 'heading', level: headingMatch[1].length as 1 | 2 | 3, content: headingMatch[2].trim() });
      i += 1;
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length) {
        const itemLine = lines[i].trim();
        if (!/^[-*]\s+/.test(itemLine)) break;
        items.push(itemLine.replace(/^[-*]\s+/, ''));
        i += 1;
      }
      blocks.push({ type: 'list', items });
      continue;
    }

    const paragraphLines: string[] = [];
    while (i < lines.length) {
      const paragraphLine = lines[i];
      const paragraphTrimmed = paragraphLine.trim();
      if (!paragraphTrimmed || paragraphTrimmed.startsWith('```') || /^(#{1,3})\s+/.test(paragraphTrimmed) || /^[-*]\s+/.test(paragraphTrimmed)) break;
      paragraphLines.push(paragraphLine);
      i += 1;
    }
    blocks.push({ type: 'paragraph', content: paragraphLines.join('\n').trim() });
  }

  return blocks;
}

function renderInline(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index));
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(<strong key={`${match.index}-bold`} className="font-semibold text-text">{token.slice(2, -2)}</strong>);
    } else if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code
          key={`${match.index}-code`}
          className="rounded px-1.5 py-0.5 font-mono text-[0.85em] text-text"
          style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
        >
          {token.slice(1, -1)}
        </code>
      );
    }
    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts;
}

function renderCodeBlock(language: string, content: string) {
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

export function MarkdownContent({ content }: MarkdownContentProps) {
  const blocks = parseBlocks(content);

  return (
    <div className="space-y-4">
      {blocks.map((block, index) => {
        if (block.type === 'heading') {
          const className =
            block.level === 1 ? 'text-lg font-semibold text-text'
            : block.level === 2 ? 'text-base font-semibold text-text'
            : 'text-sm font-semibold text-text';
          return <div key={index} className={className}>{renderInline(block.content)}</div>;
        }
        if (block.type === 'list') {
          return (
            <ul key={index} className="space-y-2 pl-5 text-sm leading-relaxed text-text-2">
              {block.items.map((item, itemIndex) => (
                <li key={`${index}-${itemIndex}`} className="list-disc">{renderInline(item)}</li>
              ))}
            </ul>
          );
        }
        if (block.type === 'code') {
          return <div key={index}>{renderCodeBlock(block.language, block.content)}</div>;
        }
        return <p key={index} className="whitespace-pre-wrap text-sm leading-relaxed text-text-2">{renderInline(block.content)}</p>;
      })}
    </div>
  );
}
