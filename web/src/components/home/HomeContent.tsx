'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { ArrowRight, Check, FlaskConical, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useLocale } from '@/context/LocaleContext';
import type { LearningPath } from '@/lib/types';

interface HomeContentProps {
  stats: { total: number; easy: number; medium: number; hard: number };
}

type PathWithProgress = LearningPath & { solved: number; total: number };

interface CategoryInfo {
  name: string;
  description: string;
  total: number;
  easy: number;
  medium: number;
  hard: number;
}

function SectionHeader({
  eyebrow,
  title,
  linkText,
  linkHref,
}: {
  eyebrow: string;
  title: string;
  linkText?: string;
  linkHref?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-6 mb-8">
      <div>
        <div className="eyebrow mb-2">{eyebrow}</div>
        <h2 className="text-[clamp(24px,2.6vw,32px)] font-semibold tracking-[-0.025em] mt-2">
          {title}
        </h2>
      </div>
      {linkText && linkHref && (
        <Link
          href={linkHref}
          className="text-[13px] text-text-2 hover:text-text inline-flex items-center gap-1.5 mono shrink-0 transition-colors"
        >
          {linkText} →
        </Link>
      )}
    </div>
  );
}

function DifficultyBars({ easy, medium, hard }: { easy: number; medium: number; hard: number }) {
  const max = Math.max(easy, medium, hard, 1);
  return (
    <span className="inline-flex gap-[2px] items-end h-[10px]">
      {[
        { n: easy, color: 'var(--easy)' },
        { n: medium, color: 'var(--medium)' },
        { n: hard, color: 'var(--hard)' },
      ].map((d, i) => (
        <span
          key={i}
          className="w-[3px] rounded-[1px]"
          style={{ height: `${Math.max(2, (d.n / max) * 10)}px`, background: d.color }}
        />
      ))}
    </span>
  );
}

const CODE_LINES = [
  { n: 1, code: '<span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))"># Implement causal self-attention.</span>', style: 'italic' },
  { n: 2, code: '<span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">import</span> <span style="font-weight:500">torch</span>' },
  { n: 3, code: '<span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">import</span> <span style="font-weight:500">torch.nn.functional</span> <span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">as</span> <span style="font-weight:500">F</span>' },
  { n: 4, code: '' },
  { n: 5, code: '<span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">def</span> <span style="font-weight:500">causal_attention</span><span style="color:var(--text-2)">(</span>q<span style="color:var(--text-2)">,</span> k<span style="color:var(--text-2)">,</span> v<span style="color:var(--text-2)">):</span>' },
  { n: 6, code: '    d <span style="color:var(--text-2)">=</span> q<span style="color:var(--text-2)">.</span>size<span style="color:var(--text-2)">(-</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">1</span><span style="color:var(--text-2)">)</span>' },
  { n: 7, code: '    scores <span style="color:var(--text-2)">=</span> q <span style="color:var(--text-2)">@</span> k<span style="color:var(--text-2)">.</span>transpose<span style="color:var(--text-2)">(-</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">2</span><span style="color:var(--text-2)">,-</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">1</span><span style="color:var(--text-2)">)</span> <span style="color:var(--text-2)">/</span> d<span style="color:var(--text-2)">**</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">0.5</span>' },
  { n: 8, code: '    T <span style="color:var(--text-2)">=</span> q<span style="color:var(--text-2)">.</span>size<span style="color:var(--text-2)">(-</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">2</span><span style="color:var(--text-2)">)</span>' },
  { n: 9, code: '    mask <span style="color:var(--text-2)">=</span> torch<span style="color:var(--text-2)">.</span>triu<span style="color:var(--text-2)">(</span>torch<span style="color:var(--text-2)">.</span>ones<span style="color:var(--text-2)">(</span>T<span style="color:var(--text-2)">,</span>T<span style="color:var(--text-2)">),</span> <span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">1</span><span style="color:var(--text-2)">).</span>bool<span style="color:var(--text-2)">()</span>' },
  { n: 10, code: '    scores <span style="color:var(--text-2)">=</span> scores<span style="color:var(--text-2)">.</span>masked_fill<span style="color:var(--text-2)">(</span>mask<span style="color:var(--text-2)">,</span> <span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">float</span><span style="color:var(--text-2)">(</span><span style="color:color-mix(in oklab,var(--easy) 70%,var(--text))">&#39;-inf&#39;</span><span style="color:var(--text-2)">))</span>' },
  { n: 11, code: '    <span style="color:color-mix(in oklab,var(--accent) 80%,var(--text))">return</span> F<span style="color:var(--text-2)">.</span>softmax<span style="color:var(--text-2)">(</span>scores<span style="color:var(--text-2)">,</span> <span style="color:var(--text-2)">-</span><span style="color:color-mix(in oklab,var(--hard) 65%,var(--text))">1</span><span style="color:var(--text-2)">)</span> <span style="color:var(--text-2)">@</span> v' },
];

const TESTS = [
  { name: 'shape_check', status: 'pass' as const, time: '3.2ms' },
  { name: 'masked_entries', status: 'pass' as const, time: '4.9ms' },
  { name: 'softmax_rows', status: 'pass' as const, time: '4.1ms' },
  { name: 'grad_flow', status: 'run' as const, time: 'running' },
];

export function HomeContent({ stats }: HomeContentProps) {
  const { locale, t } = useLocale();
  const router = useRouter();
  const [paths, setPaths] = useState<PathWithProgress[]>([]);
  const [categories, setCategories] = useState<CategoryInfo[]>([]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLElement &&
        (e.target.tagName === 'INPUT' ||
          e.target.tagName === 'TEXTAREA' ||
          e.target.isContentEditable)
      ) {
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        router.push('/paths');
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [router]);

  useEffect(() => {
    fetch('/api/paths')
      .then((r) => r.json())
      .then((d) => {
        const pathsList = d.paths ?? [];
        setPaths(pathsList);
        // Derive categories from paths (each path = a category)
        const cats: CategoryInfo[] = pathsList.map((p: PathWithProgress) => ({
          name: locale === 'zh' ? p.titleZh : p.titleEn,
          description: locale === 'zh' ? p.descriptionZh : p.descriptionEn,
          total: p.total,
          easy: Math.round(p.total * 0.25),
          medium: Math.round(p.total * 0.45),
          hard: p.total - Math.round(p.total * 0.25) - Math.round(p.total * 0.45),
        }));
        setCategories(cats);
      });
  }, [locale]);

  return (
    <main className="max-w-[1200px] mx-auto px-7">
      {/* Hero */}
      <section className="pt-20 pb-24">
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] gap-16 items-center">
          <div>
            <span
              className="inline-flex items-center gap-2 px-2.5 py-[5px] rounded-pill mono text-xs text-text-2 mb-7"
              style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
            >
              <span className="font-semibold text-text">{stats.total}</span>
              <span>{t('heroPill', { count: stats.total }).replace(`${stats.total} `, '')}</span>
              <span className="text-text-3">·</span>
              <span className="text-text-3">no GPU required</span>
            </span>

            <h1 className="text-[clamp(36px,5vw,56px)] font-semibold tracking-[-0.035em] leading-[1.05] mb-5">
              {t('heroTitle').split('\n').map((line, i) => (
                <span key={i}>
                  {i > 0 && <br />}
                  {line}
                </span>
              ))}
            </h1>

            <p className="text-base text-text-2 leading-relaxed max-w-[52ch] mb-8">
              {t('heroSubtitle')}
            </p>

            <div className="flex items-center gap-3">
              <Link href="/problems">
                <Button size="lg">
                  {t('startPracticing')}
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
              <Link href="/paths">
                <Button variant="secondary" size="lg">
                  {t('browsePaths')}
                  <span className="kbd">⌘K</span>
                </Button>
              </Link>
            </div>

            <div className="flex gap-6 mt-8 pt-5" style={{ borderTop: '1px dashed var(--line)' }}>
              {[
                { k: t('metaTotal'), v: t('metaTotalVal', { n: stats.total }) },
                { k: t('metaCoverage'), v: t('metaCoverageVal', { n: 13 }) },
                { k: t('metaRuntime'), v: t('metaRuntimeVal') },
                { k: t('metaJudge'), v: 'torch_judge' },
              ].map((m) => (
                <div key={m.k}>
                  <div className="eyebrow">{m.k}</div>
                  <div className="mono text-sm mt-1">{m.v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Editor preview with line numbers + test sidebar */}
          <div
            className="rounded-[14px] overflow-hidden hidden lg:block relative"
            style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
          >
            {/* Grid backdrop */}
            <div
              className="absolute inset-0 pointer-events-none opacity-[0.35]"
              style={{
                backgroundImage: 'radial-gradient(var(--line) 1px, transparent 1px)',
                backgroundSize: '18px 18px',
                maskImage: 'radial-gradient(ellipse at 30% 30%, black 40%, transparent 75%)',
                WebkitMaskImage: 'radial-gradient(ellipse at 30% 30%, black 40%, transparent 75%)',
              }}
            />
            {/* Editor header */}
            <div
              className="relative flex items-center gap-2.5 px-3.5 h-10 text-xs mono text-text-2"
              style={{
                borderBottom: '1px solid var(--line)',
                background: 'color-mix(in oklab, var(--text) 2%, var(--bg-elev))',
              }}
            >
              <span className="flex gap-1.5">
                <span className="w-[9px] h-[9px] rounded-full" style={{ background: 'color-mix(in oklab, var(--hard) 60%, transparent)' }} />
                <span className="w-[9px] h-[9px] rounded-full" style={{ background: 'color-mix(in oklab, var(--medium) 60%, transparent)' }} />
                <span className="w-[9px] h-[9px] rounded-full" style={{ background: 'color-mix(in oklab, var(--easy) 60%, transparent)' }} />
              </span>
              <span>
                <span className="text-text-3">attention / </span>
                <span className="text-text">causal_self_attention.py</span>
              </span>
              <span
                className="ml-auto mono text-[11px] px-2 py-[3px] rounded-[6px]"
                style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}
              >
                MEDIUM · 38%
              </span>
            </div>
            {/* Editor body: code + tests */}
            <div className="relative grid grid-cols-[1fr_280px] max-[720px]:grid-cols-1">
              {/* Code with line numbers */}
              <div className="relative p-4 overflow-hidden mono text-[12.5px] leading-[1.75]">
                {CODE_LINES.map((line) => (
                  <div key={line.n} className="grid grid-cols-[28px_1fr] gap-3.5">
                    <span className="text-right text-text-3 select-none">{line.n}</span>
                    <span
                      className="whitespace-pre"
                      style={line.style === 'italic' ? { fontStyle: 'italic' } : undefined}
                      dangerouslySetInnerHTML={{ __html: line.code || '&nbsp;' }}
                    />
                  </div>
                ))}
              </div>
              {/* Test sidebar */}
              <div
                className="flex flex-col gap-2 p-3.5 max-[720px]:hidden"
                style={{ borderLeft: '1px solid var(--line)', background: 'var(--bg-sunken)' }}
              >
                <div className="mono text-[11px] text-text-3 tracking-[0.1em] uppercase py-0.5 pb-1.5">
                  Tests · {TESTS.length}
                </div>
                {TESTS.map((test) => (
                  <div
                    key={test.name}
                    className="flex items-center gap-2.5 px-2.5 py-2 rounded-lg mono text-xs"
                    style={{ background: 'var(--bg-elev)', border: '1px solid var(--line)' }}
                  >
                    <span
                      className="w-4 h-4 rounded-[5px] inline-flex items-center justify-center text-[10px] flex-shrink-0"
                      style={
                        test.status === 'pass'
                          ? { background: 'var(--easy)', color: '#fff' }
                          : { background: 'transparent', color: 'var(--text-3)', border: '1px solid var(--line)' }
                      }
                    >
                      {test.status === 'pass' ? '✓' : '·'}
                    </span>
                    <span className="text-text">{test.name}</span>
                    <span className="ml-auto text-text-3 inline-flex items-center gap-1.5">
                      {test.time}
                      {test.status === 'run' && (
                        <span
                          className="w-1.5 h-1.5 rounded-full inline-block animate-pulse"
                          style={{ background: 'var(--accent)' }}
                        />
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Ticker */}
      <div
        className="overflow-hidden mono text-xs text-text-2"
        style={{
          padding: '14px 0',
          borderTop: '1px solid var(--line)',
          borderBottom: '1px solid var(--line)',
          maskImage: 'linear-gradient(90deg, transparent, black 10%, black 90%, transparent)',
          WebkitMaskImage: 'linear-gradient(90deg, transparent, black 10%, black 90%, transparent)',
        }}
        aria-hidden="true"
      >
        <div className="flex gap-10 shrink-0 w-max animate-ticker">
          {[0, 1].map((copy) =>
            [
              'MultiHeadAttention',
              'Flash Attention (tiled)',
              'Rotary Position Embedding',
              'DPO Loss',
              'GRPO Loss',
              'Speculative Decoding',
              'Paged Attention',
              'LoRA / QLoRA',
              'Mamba SSM',
              'Mixture of Experts',
              'FSDP Training Step',
              'Ring Attention',
              'Flow Matching',
              'adaLN-Zero',
              'Multi-Token Prediction',
            ].map((name) => (
              <span key={`${copy}-${name}`} className="inline-flex items-center gap-2 whitespace-nowrap">
                <span className="w-1 h-1 rounded-full" style={{ background: 'var(--text-3)' }} />
                {name}
              </span>
            ))
          )}
        </div>
      </div>

      {/* Categories */}
      {categories.length > 0 && (
        <section className="py-20" style={{ borderTop: '1px solid var(--line)' }}>
          <SectionHeader
            eyebrow={locale === 'zh' ? '§ 01 — 题目集' : '§ 01 — Problem set'}
            title={locale === 'zh' ? '九条路径覆盖全部核心领域。' : 'Nine paths. Every core domain covered.'}
            linkText={locale === 'zh' ? `全部 ${stats.total} 题` : `All ${stats.total} problems`}
            linkHref="/problems"
          />
          <div
            className="grid grid-cols-4 max-[960px]:grid-cols-2 max-[560px]:grid-cols-1 rounded-[14px] overflow-hidden"
            style={{ gap: '1px', background: 'var(--line)', border: '1px solid var(--line)' }}
          >
            {categories.slice(0, 8).map((cat, i) => (
              <Link
                key={cat.name}
                href={paths[i] ? `/paths/${paths[i].id}` : '/problems'}
                className="flex flex-col gap-2.5 p-[22px] min-h-[168px] relative transition-colors duration-[180ms] group"
                style={{ background: 'var(--bg-elev)' }}
                onMouseEnter={(e) => (e.currentTarget.style.background = 'color-mix(in oklab, var(--accent) 3%, var(--bg-elev))')}
                onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--bg-elev)')}
              >
                <span className="mono text-[11px] text-text-3 tracking-[0.12em]">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <h3 className="text-[17px] font-semibold tracking-[-0.012em]">{cat.name}</h3>
                <div
                  className="mt-auto flex items-center gap-2.5 pt-3"
                  style={{ borderTop: '1px dashed var(--line)' }}
                >
                  <DifficultyBars easy={cat.easy} medium={cat.medium} hard={cat.hard} />
                  <span className="mono text-xs text-text-2 tabular-nums">
                    {cat.total} {locale === 'zh' ? '题' : 'problems'}
                  </span>
                  <span className="ml-auto text-text-3 transition-[color,transform] duration-[180ms] group-hover:text-accent group-hover:translate-x-[2px]">
                    <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}

      {/* Learning Paths */}
      {paths.length > 0 && (
        <section className="py-20" style={{ borderTop: '1px solid var(--line)' }}>
          <SectionHeader
            eyebrow={locale === 'zh' ? '§ 02 — 学习路径' : '§ 02 — Learning paths'}
            title={locale === 'zh' ? '选一个方向。' : 'Pick a destination.'}
            linkText={locale === 'zh' ? `全部 ${paths.length} 条路径` : `All ${paths.length} paths`}
            linkHref="/paths"
          />
          <div className="grid grid-cols-3 max-[900px]:grid-cols-2 max-[600px]:grid-cols-1 gap-3.5">
            {paths.map((path, i) => {
              const title = locale === 'zh' ? path.titleZh : path.titleEn;
              const desc = locale === 'zh' ? path.descriptionZh : path.descriptionEn;
              const pct = path.total > 0 ? Math.round((path.solved / path.total) * 100) : 0;
              const tag = `PATH_${String(i + 1).padStart(2, '0')}`;
              return (
                <Link
                  key={path.id}
                  href={`/paths/${path.id}`}
                  className="flex flex-col gap-3.5 p-5 min-h-[172px] rounded-xl relative transition-[border-color,background] duration-150 group"
                  style={{ background: 'var(--bg-elev)', border: '1px solid var(--line)' }}
                  onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'var(--accent-line)')}
                  onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--line)')}
                >
                  <div className="flex items-center gap-2.5">
                    <span className="mono text-[10.5px] text-text-3 tracking-[0.12em]">{tag}</span>
                  </div>
                  <h3 className="text-[15.5px] font-semibold tracking-[-0.012em]">{title}</h3>
                  <p className="text-[13px] text-text-2 leading-relaxed">{desc}</p>
                  <div
                    className="mt-auto flex items-center gap-2.5 mono text-[11.5px] text-text-2"
                  >
                    <span>{Math.round(pct / 100 * path.total)}/{path.total}</span>
                    <div className="flex-1 h-[3px] rounded-pill relative" style={{ background: 'var(--line)' }}>
                      <div
                        className="absolute inset-0 rounded-pill"
                        style={{ width: `${pct}%`, background: 'var(--accent)' }}
                      />
                    </div>
                    <span className="tabular-nums">{pct}%</span>
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      )}

      {/* Features */}
      <section className="py-20" style={{ borderTop: '1px solid var(--line)' }}>
        <SectionHeader
          eyebrow={locale === 'zh' ? '§ 03 — 工作方式' : '§ 03 — How it works'}
          title={locale === 'zh' ? '读论文，然后写代码。' : 'Read the paper, then write the code.'}
        />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { num: '01', icon: Check, title: t('feat1Title'), desc: t('feat1Desc') },
            { num: '02', icon: FlaskConical, title: t('feat2Title'), desc: t('feat2Desc') },
            { num: '03', icon: BarChart3, title: t('feat3Title'), desc: t('feat3Desc') },
          ].map((f) => (
            <div key={f.num} className="pt-5" style={{ borderTop: '1px solid var(--line)' }}>
              <div className="flex items-center gap-3 mb-4">
                <div className="mono text-[11px] text-text-3 tracking-[0.12em]">{f.num}</div>
                <div
                  className="w-7 h-7 rounded-lg inline-flex items-center justify-center text-text-2"
                  style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}
                >
                  <f.icon className="w-3.5 h-3.5" strokeWidth={1.6} />
                </div>
              </div>
              <h4 className="text-[15.5px] font-semibold tracking-[-0.01em] mb-1.5">{f.title}</h4>
              <p className="text-[13.5px] text-text-2 leading-relaxed max-w-[40ch]">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
