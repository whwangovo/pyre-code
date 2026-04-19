'use client';

import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useLocale } from '@/context/LocaleContext';

interface HomeContentProps {
  stats: { total: number; easy: number; medium: number; hard: number };
}

export function HomeContent({ stats }: HomeContentProps) {
  const { t } = useLocale();

  return (
    <main className="max-w-[1280px] mx-auto px-7">
      {/* Hero */}
      <section className="pt-20 pb-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <div>
            <span
              className="inline-flex items-center gap-2 px-2.5 py-[5px] rounded-pill mono text-xs text-text-2 mb-7"
              style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
            >
              <span className="font-semibold text-text">{stats.total}</span>
              <span>{t('heroPill', { count: stats.total }).replace(`${stats.total} `, '')}</span>
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

          {/* Editor preview */}
          <div
            className="rounded-[14px] overflow-hidden hidden lg:block"
            style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
          >
            <div
              className="flex items-center gap-3 px-4 h-10 text-xs mono text-text-2"
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
              <span className="text-text">causal_self_attention.py</span>
            </div>
            <pre className="p-4 text-[13px] leading-[1.75] mono text-text-2 overflow-hidden">
{`# Implement causal self-attention.
import torch
import torch.nn.functional as F

def causal_attention(q, k, v):
    # (B, H, T, D) → (B, H, T, D)
    d = q.size(-1)
    T = q.size(-2)
    scores = q @ k.transpose(-2,-1) / d**0.5
    mask = torch.triu(torch.ones(T, T), 1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, -1)
    return attn @ v`}
            </pre>
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

      {/* Difficulty stats */}
      <section style={{ borderTop: '1px solid var(--line)' }}>
        <div className="grid grid-cols-3 gap-6 py-6">
          {[
            { label: t('Easy'), count: stats.easy, color: 'var(--easy)' },
            { label: t('Medium'), count: stats.medium, color: 'var(--medium)' },
            { label: t('Hard'), count: stats.hard, color: 'var(--hard)' },
          ].map((s) => (
            <div key={s.label} className="text-center">
              <div className="text-[28px] font-semibold tracking-tight" style={{ color: s.color }}>{s.count}</div>
              <div className="eyebrow mt-1">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section style={{ borderTop: '1px solid var(--line)' }}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 py-6">
          {[
            { num: '01', title: t('feat1Title'), desc: t('feat1Desc') },
            { num: '02', title: t('feat2Title'), desc: t('feat2Desc') },
            { num: '03', title: t('feat3Title'), desc: t('feat3Desc') },
          ].map((f) => (
            <div key={f.num} className="pt-5" style={{ borderTop: '1px solid var(--line)' }}>
              <div className="mono text-[11px] text-text-3 tracking-[0.12em]">{f.num}</div>
              <h4 className="text-[15.5px] font-semibold tracking-[-0.01em] mt-4 mb-1.5">{f.title}</h4>
              <p className="text-[13.5px] text-text-2 leading-relaxed max-w-[40ch]">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
