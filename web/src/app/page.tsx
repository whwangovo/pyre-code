import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { HomeContent } from '@/components/home/HomeContent';
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

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <HomeContent stats={stats} />
      <Footer />
    </div>
  );
}
