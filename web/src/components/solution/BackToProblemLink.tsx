'use client';

import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { useLocale } from '@/context/LocaleContext';

export function BackToProblemLink({ id }: { id: string }) {
  const { t } = useLocale();
  return (
    <Link
      href={`/problems/${id}`}
      className="inline-flex items-center gap-1.5 text-sm text-text-2 hover:text-accent transition-colors mb-6"
    >
      <ArrowLeft className="w-4 h-4" />
      {t('backToProblem')}
    </Link>
  );
}
