'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { MarkdownContent } from './MarkdownContent';
import { useLocale } from '@/context/LocaleContext';
import type { KnowledgeEntry as Entry, KnowledgeTopic as Topic } from '@/lib/knowledge';

// The tab unmounts when hidden; keep each note so switching back doesn't refetch
const cache = new Map<string, Promise<Entry | null>>();

function fetchEntry(problemId: string): Promise<Entry | null> {
  let pending = cache.get(problemId);
  if (!pending) {
    pending = fetch(`/api/knowledge/${encodeURIComponent(problemId)}`)
      .then((res) => (res.ok ? res.json() : null));
    pending.catch(() => cache.delete(problemId));
    cache.set(problemId, pending);
  }
  return pending;
}

const segmentClass =
  'px-3 h-[26px] rounded-[6px] text-[13px] text-text-2 cursor-pointer transition-[background,color] duration-150 data-[active=true]:bg-[color-mix(in_oklab,var(--text)_7%,transparent)] data-[active=true]:text-text';

function TopicProblems({ topic, problemId }: { topic: Topic; problemId: string }) {
  const { t, locale } = useLocale();
  return (
    <div className="mb-6 rounded-xl px-4 py-3" style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}>
      <div className="eyebrow mb-2">{t('topicProblems')}</div>
      <ol className="flex flex-wrap gap-x-5 gap-y-1.5 text-sm">
        {topic.problems.map((p, i) => {
          const title = locale === 'zh' ? p.titleZh : p.title;
          return (
            <li key={p.id} className="flex items-baseline gap-1.5">
              <span className="mono text-[11px] text-text-3 tabular-nums">{i + 1}</span>
              {p.id === problemId ? (
                <span className="font-medium text-text">{title}</span>
              ) : (
                <Link href={`/problems/${p.id}`} className="text-accent hover:underline underline-offset-2">{title}</Link>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

interface KnowledgeTabProps {
  problemId: string;
}

export function KnowledgeTab({ problemId }: KnowledgeTabProps) {
  const { t, locale } = useLocale();
  const [loaded, setLoaded] = useState<{ id: string; entry: Entry | null } | null>(null);
  const [showTopic, setShowTopic] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchEntry(problemId)
      .then((entry) => !cancelled && setLoaded({ id: problemId, entry }))
      .catch(() => !cancelled && setLoaded({ id: problemId, entry: null }));
    return () => {
      cancelled = true;
    };
  }, [problemId]);

  if (loaded?.id !== problemId) {
    return <div className="p-6 text-sm text-text-3">{t('loadingKnowledge')}</div>;
  }
  if (!loaded.entry) {
    return <div className="p-6 text-sm text-text-3">{t('noKnowledge')}</div>;
  }
  const { note, topic } = loaded.entry;
  if (!topic) {
    return (
      <div className="px-7 py-6">
        <MarkdownContent content={note[locale] ?? note.zh} />
      </div>
    );
  }
  const topicTitle = locale === 'zh' ? topic.titleZh : topic.titleEn;
  return (
    <div className="px-7 py-6">
      <div className="flex gap-0.5 mb-5">
        <button type="button" data-active={!showTopic} className={segmentClass} onClick={() => setShowTopic(false)}>
          {t('knowledgeThisProblem')}
        </button>
        <button type="button" data-active={showTopic} className={segmentClass} onClick={() => setShowTopic(true)}>
          {t('knowledgeTopic')} · {topicTitle}
        </button>
      </div>
      {showTopic ? (
        <>
          <TopicProblems topic={topic} problemId={problemId} />
          <MarkdownContent content={topic.overview[locale] ?? topic.overview.zh} />
        </>
      ) : (
        <MarkdownContent content={note[locale] ?? note.zh} />
      )}
    </div>
  );
}
