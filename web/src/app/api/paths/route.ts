import { NextResponse } from 'next/server';
import pathsData from '@/lib/paths.json';
import problemsData from '@/lib/problems.json';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import { getSessionToken } from '@/lib/session';
import type { LearningPath } from '@/lib/types';

const difficultyOf = new Map(
  (problemsData.problems as Array<{ id: string; difficulty: string }>).map((p) => [p.id, p.difficulty]),
);

export async function GET() {
  const sessionToken = await getSessionToken();

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

  const paths = (pathsData.paths as LearningPath[]).map((path) => {
    const solved = path.problems.filter((id) => progressMap[id]?.status === 'solved').length;
    const count = (difficulty: string) => path.problems.filter((id) => difficultyOf.get(id) === difficulty).length;
    return {
      ...path,
      solved,
      total: path.problems.length,
      easy: count('Easy'),
      medium: count('Medium'),
      hard: count('Hard'),
    };
  });

  return NextResponse.json({ paths });
}
