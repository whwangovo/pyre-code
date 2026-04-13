import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { GRADING_SERVICE_URL } from '@/lib/constants';

export async function GET(_req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get('session_token')?.value;
  if (!sessionToken) return NextResponse.json([]);

  const userRes = await fetch(`${GRADING_SERVICE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken }),
  });
  if (!userRes.ok) return NextResponse.json([]);

  const { userId } = await userRes.json();
  const res = await fetch(`${GRADING_SERVICE_URL}/submissions/${userId}/${taskId}`);
  if (!res.ok) return NextResponse.json([]);
  return NextResponse.json(await res.json());
}
