'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Flame, SwatchBook } from 'lucide-react';
import { GlassBar } from '@/components/ui/GlassBar.classic';
import { cn } from '@/lib/utils';
import { useLocale } from '@/context/LocaleContext';
import { useDesign } from '@/context/DesignContext';

interface TopNavProps {
  solvedCount?: number;
  totalCount?: number;
}

export function TopNav({ solvedCount, totalCount }: TopNavProps) {
  const pathname = usePathname();
  const { locale, setLocale, t } = useLocale();
  const { toggleDesign } = useDesign();

  const links = [
    { href: '/', label: t('home') },
    { href: '/problems', label: t('problems') },
    { href: '/paths', label: t('paths') },
  ];

  return (
    <GlassBar as="nav" className="px-6 h-14 flex items-center justify-between">
      <div className="flex items-center gap-8">
        <Link href="/" className="flex items-center gap-2 text-text-primary font-semibold text-lg tracking-tight">
          <Flame className="w-5 h-5 text-accent" />
          Pyre Code
        </Link>
        <div className="flex items-center gap-1">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                'px-3 py-1.5 rounded-lg text-sm transition-colors duration-200',
                pathname === link.href
                  ? 'text-accent font-medium bg-accent/5'
                  : 'text-text-secondary hover:text-text-primary hover:bg-gray-100'
              )}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-3">
        {solvedCount !== undefined && totalCount !== undefined && (
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <Flame className="w-4 h-4 text-medium" />
            <span>{t('solvedCount', { solved: solvedCount, total: totalCount })}</span>
          </div>
        )}
        <button
          onClick={toggleDesign}
          className="px-2.5 py-1 rounded-lg text-xs font-medium text-text-secondary hover:text-text-primary hover:bg-gray-100 transition-colors"
          title="Switch design"
        >
          <SwatchBook className="w-4 h-4" />
        </button>
        <button
          onClick={() => setLocale(locale === 'en' ? 'zh' : 'en')}
          className="px-2.5 py-1 rounded-lg text-xs font-medium text-text-secondary hover:text-text-primary hover:bg-gray-100 transition-colors"
        >
          {locale === 'en' ? '中文' : 'EN'}
        </button>
      </div>
    </GlassBar>
  );
}
