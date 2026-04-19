import { cn } from '@/lib/utils';

interface GlassBarProps {
  children: React.ReactNode;
  className?: string;
  as?: 'nav' | 'div' | 'footer';
}

export function GlassBar({ children, className, as: Tag = 'div' }: GlassBarProps) {
  return (
    <Tag
      className={cn('sticky top-0 z-50', className)}
      style={{
        backdropFilter: 'saturate(180%) blur(14px)',
        WebkitBackdropFilter: 'saturate(180%) blur(14px)',
        background: 'color-mix(in oklab, var(--bg) 82%, transparent)',
        borderBottom: '1px solid var(--line)',
      }}
    >
      {children}
    </Tag>
  );
}
