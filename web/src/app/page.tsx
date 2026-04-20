'use client';

import { TopNav } from '@/components/layout/TopNav';
import { TopNav as TopNavClassic } from '@/components/layout/TopNav.classic';
import { Footer } from '@/components/layout/Footer';
import { HomeContent } from '@/components/home/HomeContent';
import { HomeContentClassic } from '@/components/home/HomeContent.classic';
import { useDesign } from '@/context/DesignContext';
import problems from '@/lib/problems.json';

function getStats() {
  const list = problems.problems;
  return {
    total: list.length,
    easy: list.filter((p) => p.difficulty === 'Easy').length,
    medium: list.filter((p) => p.difficulty === 'Medium').length,
    hard: list.filter((p) => p.difficulty === 'Hard').length,
  };
}

export default function HomePage() {
  const stats = getStats();
  const { design } = useDesign();

  if (design === 'classic') {
    return (
      <div className="min-h-screen bg-surface">
        <TopNavClassic />
        <HomeContentClassic stats={stats} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <HomeContent stats={stats} />
      <Footer />
    </div>
  );
}
