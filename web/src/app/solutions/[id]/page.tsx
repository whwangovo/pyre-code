'use client';

import { use } from 'react';
import Link from 'next/link';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { SolutionPageContent } from '@/components/solution/SolutionPage';
import { SolutionPageClassic } from '@/components/solution/SolutionPageRoute.classic';
import { useDesign } from '@/context/DesignContext';

export default function SolutionsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { design } = useDesign();

  if (design === 'classic') return <SolutionPageClassic id={id} />;

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="max-w-[1280px] mx-auto px-7 pt-8 pb-20">
        <SolutionPageContent problemId={id} />
      </main>
      <Footer />
    </div>
  );
}
