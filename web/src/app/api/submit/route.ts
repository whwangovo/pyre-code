import { NextResponse } from 'next/server';
import { getSessionToken } from '@/lib/session';
import { gradingFetch } from '@/lib/grading';
import type { SubmissionResult } from '@/lib/types';

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as { taskId?: unknown; code?: unknown; mode?: unknown } | null;
  const { taskId, code, mode } = body ?? {};
  if (typeof taskId !== 'string' || typeof code !== 'string') {
    return NextResponse.json({ error: 'Missing taskId or code' }, { status: 400 });
  }

  const existingToken = await getSessionToken();
  const sessionToken = existingToken ?? crypto.randomUUID();

  try {
    // One backend call: grading, user creation, progress and submission history are
    // saved together in a single transaction there.
    const res = await gradingFetch(sessionToken, '/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionToken, taskId, code, mode: mode === 'acm' ? 'acm' : 'core' }),
    });
    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json(
        { passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: errText },
        { status: 502 }
      );
    }
    const result: SubmissionResult = await res.json();
    const response = NextResponse.json(result);
    if (!existingToken) {
      response.cookies.set('session_token', sessionToken, { httpOnly: true, maxAge: 60 * 60 * 24 * 30 });
    }
    return response;
  } catch {
    return NextResponse.json(
      { passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: 'Grading service unreachable' },
      { status: 502 }
    );
  }
}
