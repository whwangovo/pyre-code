import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import pathsData from '@/lib/paths.json';
import problemsData from '@/lib/problems.json';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import type { LearningPath } from '@/lib/types';

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const path = (pathsData.paths as LearningPath[]).find((p) => p.id === id);
  if (!path) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  const cookieStore = await cookies();
  const sessionToken = cookieStore.get('session_token')?.value;

  let progressMap: Record<string, { status: string }> = {};
  if (sessionToken) {
    try {
      const userRes = await fetch(`${GRADING_SERVICE_URL}/users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionToken }),
      });
      if (userRes.ok) {
        const { userId } = await userRes.json();
        const progressRes = await fetch(`${GRADING_SERVICE_URL}/progress/${userId}`);
        if (progressRes.ok) progressMap = await progressRes.json();
      }
    } catch {
      // continue without progress
    }
  }

  const problems = path.problems.map((problemId) => {
    const problem = (problemsData as { problems: Array<{ id: string; title: string; titleZh: string; difficulty: string }> }).problems.find((p) => p.id === problemId);
    return {
      id: problemId,
      title: problem?.title ?? problemId,
      titleZh: problem?.titleZh ?? problemId,
      difficulty: (problem?.difficulty ?? 'Easy') as 'Easy' | 'Medium' | 'Hard',
      status: (progressMap[problemId]?.status ?? 'todo') as 'todo' | 'attempted' | 'solved',
    };
  });

  const solved = problems.filter((p) => p.status === 'solved').length;

  return NextResponse.json({ ...path, problems, solved, total: path.problems.length });
}
