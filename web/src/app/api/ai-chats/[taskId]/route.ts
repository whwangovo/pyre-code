import { NextResponse } from 'next/server';
import { getSessionToken } from '@/lib/session';
import { gradingFetch } from '@/lib/grading';

/** The learner's saved AI help conversation for one problem ([] for none). */
export async function GET(_req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const sessionToken = await getSessionToken();
  if (!sessionToken) return NextResponse.json([]);

  try {
    const res = await gradingFetch(sessionToken, `/ai-chats/${encodeURIComponent(taskId)}`);
    if (!res.ok) return NextResponse.json([], { status: 502 });
    return NextResponse.json(await res.json());
  } catch {
    return NextResponse.json([], { status: 502 });
  }
}

export async function PUT(req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const { messages } = await req.json().catch(() => ({}));
  if (!Array.isArray(messages)) return NextResponse.json({ error: 'Missing messages' }, { status: 400 });

  // Chatting can be a learner's first action; the grading service creates the
  // anonymous session on the first write.
  const existingToken = await getSessionToken();
  const sessionToken = existingToken ?? crypto.randomUUID();

  try {
    const res = await gradingFetch(sessionToken, '/ai-chats', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionToken, taskId, messages }),
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
