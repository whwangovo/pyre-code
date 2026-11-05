import { NextResponse } from 'next/server';
import { getSolutionCode } from '@/lib/solutions';
import type { AiHelpRequest } from '@/lib/types';

/* ── env defaults ── */
const ENV_BASE_URL = process.env.AI_HELP_BASE_URL?.trim() || '';
const ENV_API_KEY = process.env.AI_HELP_API_KEY?.trim() || '';
const ENV_MODEL = process.env.AI_HELP_MODEL?.trim() || '';

/* ── simple in-memory rate limiter ── */
const RATE_WINDOW_MS = 60_000;
const RATE_MAX = 10;
const hits = new Map<string, number[]>();

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  // Opportunistic sweep: without it, every address that ever posted would keep a key forever
  if (hits.size > 512) {
    for (const [key, times] of hits) {
      if (times.every((at) => now - at >= RATE_WINDOW_MS)) hits.delete(key);
    }
  }
  const timestamps = (hits.get(ip) ?? []).filter((t) => now - t < RATE_WINDOW_MS);
  if (timestamps.length >= RATE_MAX) {
    hits.set(ip, timestamps);
    return true;
  }
  timestamps.push(now);
  hits.set(ip, timestamps);
  return false;
}

/* ── helpers ── */
function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, '');
}

function extractMessageContent(content: unknown): string {
  if (typeof content === 'string') return content.trim();
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object' && 'type' in item && 'text' in item) {
          const part = item as { type?: string; text?: string };
          return part.type === 'text' && part.text ? part.text : '';
        }
        return '';
      })
      .join('\n')
      .trim();
  }
  return '';
}

const MAX_HISTORY_MESSAGES = 20;

/* Problem context lives in the system prompt and is rebuilt on every turn, so the model
   always sees the learner's latest code rather than whatever they had when the chat began. */
function buildSystemPrompt(request: AiHelpRequest) {
  const testsText = request.sampleTests
    .map((test, index) => `Sample Test ${index + 1}: ${test.name}\n${test.code}`)
    .join('\n\n');
  const userCode = request.userCode?.trim();
  const isHot100 = request.track === 'hot100';

  const intro = isHot100
    ? [
        'You are chatting with someone preparing for coding interviews on Pyre Code by working through the LeetCode Hot 100 problems in Python. Think of yourself as an experienced engineer sitting next to them: talk naturally, answer what they actually ask, and follow the conversation wherever it goes: the approach, why it works, complexity, edge cases, alternative solutions, how to explain it in an interview, or their code.',
        request.mode === 'acm'
          ? 'They are answering in ACM mode: a complete program that reads stdin and prints to stdout in the exact format below, so input parsing and output formatting are part of the problem.'
          : 'They are answering in core-code mode, as on LeetCode: implement the method (or design class) only; the judge handles input and output, and List, Optional, ListNode and TreeNode are predefined.',
      ]
    : ['You are chatting with someone practicing on Pyre Code, a site where people implement the building blocks of modern AI and LLMs from scratch in PyTorch. Think of yourself as an experienced colleague sitting next to them: talk naturally, answer what they actually ask, and follow the conversation wherever it goes: concepts, math intuition, papers, how real systems do it, or their code.'];

  return [
    ...intro,
    '',
    'A few things to keep in mind:',
    '- Talk like a person, not a report. Short questions get short answers; use headings or lists only when they genuinely help.',
    '- They are here to learn by doing, so while they are working on the problem, lean toward explanations and hints that let them get there themselves. If they explicitly ask for the full solution, it is fine to give it.',
    '- The problem, a reference solution and their current code are below as background. Bring the code up when it is relevant to what they asked; do not turn every reply into a code review.',
    '- When you do point out a problem in their code, be specific about which line and why.',
    '- The chat renders Markdown (including tables) and LaTeX: use $...$ for inline math and $$...$$ on their own lines for display math; keep code in fenced code blocks.',
    `- Reply in the language the learner writes in (default: ${request.locale === 'zh' ? 'Chinese' : 'English'}).`,
    '',
    '## Problem',
    `ID: ${request.problemId}`,
    `Title: ${request.problemTitle}`,
    `${isHot100 ? 'Method / class' : 'Target function'}: ${request.functionName}`,
    '',
    request.description,
    '',
    '## Sample tests',
    testsText || 'No sample tests available.',
    '',
    '## Reference solution (the learner can also open it in the Solution tab)',
    // Looked up here rather than sent by the browser, which no longer loads every solution
    getSolutionCode(request.problemId, isHot100 ? request.mode ?? 'core' : 'core') || 'No reference solution available.',
    '',
    '## Learner\'s current code (latest snapshot; it may have changed since earlier messages)',
    request.userCode === undefined
      ? 'The learner chose not to share their code.'
      : userCode || 'The editor is empty.',
  ].join('\n');
}

function sanitizeMessages(messages: AiHelpRequest['messages'] | undefined) {
  return (Array.isArray(messages) ? messages : [])
    .filter((m) => (m?.role === 'user' || m?.role === 'assistant') && typeof m.content === 'string' && m.content.trim())
    .slice(-MAX_HISTORY_MESSAGES)
    .map((m) => ({ role: m.role, content: m.content }));
}

/* Turns an OpenAI-compatible SSE stream into a plain-text stream of content deltas. */
function sseToTextStream(body: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = '';

  const emitLine = (line: string, controller: TransformStreamDefaultController<Uint8Array>) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return;
    const data = trimmed.slice(5).trim();
    if (!data || data === '[DONE]') return;
    try {
      const delta = JSON.parse(data)?.choices?.[0]?.delta?.content;
      // Raw string deltas must not be trimmed: token boundaries often fall on spaces
      const text = typeof delta === 'string' ? delta : extractMessageContent(delta);
      if (text) controller.enqueue(encoder.encode(text));
    } catch {}
  };

  return body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) emitLine(line, controller);
    },
    flush(controller) {
      buffer += decoder.decode();
      if (buffer) emitLine(buffer, controller);
    },
  }));
}

export async function POST(request: Request) {
  const ip = request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || 'unknown';
  if (isRateLimited(ip)) {
    return NextResponse.json(
      { error: 'Too many requests. Please wait a moment before trying again.' },
      { status: 429 },
    );
  }

  const body = (await request.json()) as AiHelpRequest;

  // Server env takes priority; fall back to client config
  const baseUrl = ENV_BASE_URL || body.config?.baseUrl?.trim() || '';
  const apiKey = ENV_API_KEY || body.config?.apiKey?.trim() || '';
  const model = ENV_MODEL || body.config?.model?.trim() || '';

  if (!baseUrl || !apiKey || !model) {
    return NextResponse.json({ error: 'Missing baseUrl, apiKey, or model.' }, { status: 400 });
  }

  const messages = sanitizeMessages(body.messages);
  if (messages.at(-1)?.role !== 'user') {
    return NextResponse.json({ error: 'The last message must come from the user.' }, { status: 400 });
  }

  // No `temperature`: some models (reasoning models, newer Claude) reject non-default values
  const payload = {
    model,
    stream: true,
    messages: [{ role: 'system', content: buildSystemPrompt(body) }, ...messages],
  };

  try {
    const upstream = await fetch(`${normalizeBaseUrl(baseUrl)}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
      // Stop generating upstream when the learner hits "stop" or leaves
      signal: request.signal,
    });

    const textHeaders = { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-cache' };
    if (upstream.ok && upstream.body && upstream.headers.get('content-type')?.includes('text/event-stream')) {
      return new Response(sseToTextStream(upstream.body), { headers: textHeaders });
    }

    // Error, or an endpoint that ignores `stream: true` and answers with plain JSON
    const data = await upstream.json().catch(() => null);
    if (!upstream.ok) {
      if (upstream.status === 401) {
        return NextResponse.json({ error: 'Invalid API key. Please check your configuration.' }, { status: 401 });
      }
      if (upstream.status === 429) {
        return NextResponse.json({ error: 'API rate limit exceeded. Please try again later.' }, { status: 429 });
      }
      const errorMessage =
        data && typeof data === 'object' && 'error' in data && data.error && typeof data.error === 'object' && 'message' in data.error
          ? String(data.error.message)
          : `Upstream request failed with status ${upstream.status}.`;
      return NextResponse.json({ error: errorMessage }, { status: 502 });
    }

    const reply = extractMessageContent(data?.choices?.[0]?.message?.content);
    if (!reply) {
      return NextResponse.json({ error: 'Model returned an empty response.' }, { status: 502 });
    }

    return new Response(reply, { headers: textHeaders });
  } catch {
    return NextResponse.json({ error: 'Failed to reach the configured AI endpoint.' }, { status: 502 });
  }
}
