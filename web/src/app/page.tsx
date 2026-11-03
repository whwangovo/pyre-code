import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { HomeContent } from '@/components/home/HomeContent';
import problems from '@/lib/problems.json';
import knowledge from '@/lib/knowledge.json';

// A server component: the catalog JSON (cases included) stays on the server and only the
// counts reach the browser
function getStats() {
  // The home page presents the AI-systems track; Hot 100 has its own section
  const list = problems.problems.filter((p) => !('track' in p) || p.track !== 'hot100');
  return {
    total: list.length,
    easy: list.filter((p) => p.difficulty === 'Easy').length,
    medium: list.filter((p) => p.difficulty === 'Medium').length,
    hard: list.filter((p) => p.difficulty === 'Hard').length,
    // Concept topics, each owning a disjoint set of AI problems (see knowledge/topics.json)
    topics: Object.keys(knowledge.topics).length,
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
