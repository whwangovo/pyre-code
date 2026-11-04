import { NextResponse } from 'next/server';
import { getSessionToken } from '@/lib/session';
import { gradingFetch } from '@/lib/grading';

export async function GET(_req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const sessionToken = await getSessionToken();
  if (!sessionToken) return NextResponse.json([]);

  try {
    const res = await gradingFetch(sessionToken, `/submissions/${encodeURIComponent(taskId)}`);
    if (!res.ok) return NextResponse.json([]);
    return NextResponse.json(await res.json());
  } catch {
    return NextResponse.json([]);
  }
}
