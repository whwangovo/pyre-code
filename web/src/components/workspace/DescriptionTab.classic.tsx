'use client';

import { useState } from 'react';
import { Lightbulb, ChevronDown, ChevronRight } from 'lucide-react';
import { DifficultyBadge } from '@/components/problem/DifficultyBadge.classic';
import { useLocale } from '@/context/LocaleContext';
import type { Problem } from '@/lib/types';

interface DescriptionTabProps {
  problem: Problem;
}

/** Parse inline markdown: **bold**, `code` */
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
      parts.push(<strong key={key++} className="font-semibold text-text-primary">{boldMatch[1]}</strong>);
      remaining = remaining.slice(boldIdx + boldMatch[0].length);
    } else if (codeMatch) {
      if (codeIdx! > 0) parts.push(remaining.slice(0, codeIdx!));
      parts.push(<code key={key++} className="px-1 py-0.5 rounded bg-surface-secondary text-accent text-xs font-mono">{codeMatch[1]}</code>);
      remaining = remaining.slice(codeIdx! + codeMatch[0].length);
    }
  }
  return parts;
}

/** Render simple markdown: **bold**, `code`, - lists, blank lines */
function renderDescription(text: string) {
  return text.split('\n').map((line, i) => {
    if (line === '') return <div key={i} className="h-2" />;
    if (line.startsWith('- ')) {
      return <li key={i} className="ml-4 list-disc text-sm text-text-secondary leading-relaxed">{parseInline(line.slice(2))}</li>;
    }
    return <p key={i} className="text-sm text-text-secondary leading-relaxed">{parseInline(line)}</p>;
  });
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
          <h1 className="text-xl font-semibold tracking-tight text-text-primary">
            {problem.title}
          </h1>
          <DifficultyBadge difficulty={problem.difficulty} />
        </div>
        <p className="text-sm text-text-secondary">
          {t('implementFn', { fn: problem.functionName })}
        </p>
      </div>

      {description && (
        <div className="space-y-1">
          {renderDescription(description)}
        </div>
      )}

      {hint && (
        <div>
          <button
            onClick={() => setHintOpen(!hintOpen)}
            className="flex items-center gap-2 text-sm text-text-secondary hover:text-accent transition-colors"
          >
            <Lightbulb className="w-4 h-4" />
            <span>{t('hint')}</span>
            {hintOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>
          {hintOpen && (
            <div className="mt-2 p-3 rounded-lg bg-medium/5 text-sm text-text-secondary leading-relaxed space-y-1">
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
