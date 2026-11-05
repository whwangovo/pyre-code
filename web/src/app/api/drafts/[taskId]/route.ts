import { NextResponse } from 'next/server';
import { getSessionToken } from '@/lib/session';
import { gradingFetch } from '@/lib/grading';

/** The learner's server-side draft for one problem and answer mode, or null. */
export async function GET(req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const mode = new URL(req.url).searchParams.get('mode') ?? 'core';
  const sessionToken = await getSessionToken();
  if (!sessionToken) return NextResponse.json(null);

  try {
    const res = await gradingFetch(sessionToken, `/drafts/${encodeURIComponent(taskId)}?mode=${encodeURIComponent(mode)}`);
    if (!res.ok) return NextResponse.json(null, { status: 502 });
    return NextResponse.json(await res.json());
  } catch {
    return NextResponse.json(null, { status: 502 });
  }
}

export async function PUT(req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const { code, mode, updatedAt } = await req.json().catch(() => ({}));
  if (typeof code !== 'string' || typeof updatedAt !== 'number') {
    return NextResponse.json({ error: 'Missing code or updatedAt' }, { status: 400 });
  }

  // A draft can be a learner's first action; the grading service creates the
  // anonymous session on the first write.
  const existingToken = await getSessionToken();
  const sessionToken = existingToken ?? crypto.randomUUID();

  try {
    const res = await gradingFetch(sessionToken, '/drafts', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionToken, taskId, mode: mode ?? 'core', code, updatedAt: Math.round(updatedAt) }),
    });
    if (!res.ok) return NextResponse.json({ error: await res.text() }, { status: res.status });

    const response = NextResponse.json({ ok: true });
    if (!existingToken) {
      response.cookies.set('session_token', sessionToken, { httpOnly: true, maxAge: 60 * 60 * 24 * 30 });
    }
    return response;
  } catch {
    return NextResponse.json({ error: 'Grading service unreachable' }, { status: 502 });
  }
}
