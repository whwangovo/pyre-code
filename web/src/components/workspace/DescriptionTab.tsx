'use client';

import { useState } from 'react';
import { Lightbulb, ChevronDown, ChevronRight } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { useLocale } from '@/context/LocaleContext';
import type { Problem } from '@/lib/types';

function parseInline(text: string): (string | JSX.Element)[] {
  const parts: (string | JSX.Element)[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    const codeMatch = remaining.match(/`(.+?)`/);
    const boldIdx = boldMatch?.index ?? Infinity;
    const codeIdx = codeMatch?.index ?? Infinity;

    if (boldIdx === Infinity && codeIdx === Infinity) {
      parts.push(remaining);
      break;
    }

    if (boldIdx <= codeIdx && boldMatch) {
      if (boldIdx > 0) parts.push(remaining.slice(0, boldIdx));
      parts.push(<strong key={key++} className="font-medium text-text">{boldMatch[1]}</strong>);
      remaining = remaining.slice(boldIdx + boldMatch[0].length);
    } else if (codeMatch) {
      if (codeIdx! > 0) parts.push(remaining.slice(0, codeIdx!));
      parts.push(
        <code
          key={key++}
          className="mono text-[12.5px] px-[5px] py-px rounded text-text"
          style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
        >
          {codeMatch[1]}
        </code>
      );
      remaining = remaining.slice(codeIdx! + codeMatch[0].length);
    }
  }
  return parts;
}

function renderDescription(text: string) {
  return text.split('\n').map((line, i) => {
    if (line === '') return <div key={i} className="h-2" />;
    if (line.startsWith('- ')) {
      return <li key={i} className="ml-4 list-disc text-sm text-text-2 leading-relaxed">{parseInline(line.slice(2))}</li>;
    }
    return <p key={i} className="text-sm text-text-2 leading-relaxed">{parseInline(line)}</p>;
  });
}

interface DescriptionTabProps {
  problem: Problem;
}

export function DescriptionTab({ problem }: DescriptionTabProps) {
  const [hintOpen, setHintOpen] = useState(false);
  const { locale, t } = useLocale();

  const description = locale === 'zh' ? problem.descriptionZh : problem.descriptionEn;
  const hint = locale === 'zh' && problem.hintZh ? problem.hintZh : problem.hint;

  return (
    <div className="p-6 space-y-6">
      <div>
        <div className="flex items-center gap-3 mb-2">
          <h1 className="text-xl font-semibold tracking-tight">{problem.title}</h1>
          <Badge variant={problem.difficulty.toLowerCase() as 'easy' | 'medium' | 'hard'}>
            {problem.difficulty.toUpperCase()}
          </Badge>
        </div>
        <p className="text-sm text-text-2">{t('implementFn', { fn: problem.functionName })}</p>
      </div>

      {description && (
        <div className="space-y-1">{renderDescription(description)}</div>
      )}

      {hint && (
        <div>
          <button
            onClick={() => setHintOpen(!hintOpen)}
            className="flex items-center gap-2 text-sm text-text-2 hover:text-accent transition-colors"
          >
            <Lightbulb className="w-4 h-4" />
            <span>{t('hint')}</span>
            {hintOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>
          {hintOpen && (
            <div
              className="mt-2 p-3 rounded-lg text-sm text-text-2 leading-relaxed space-y-1"
              style={{ background: 'color-mix(in oklab, var(--medium) 5%, var(--bg-elev))' }}
            >
              {hint.split('\n').map((line, i) => (
                <p key={i}>{parseInline(line)}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
