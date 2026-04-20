import { cn } from '@/lib/utils';

interface GlassBarProps {
  children: React.ReactNode;
  className?: string;
  as?: 'nav' | 'div' | 'footer';
}

export function GlassBar({ children, className, as: Tag = 'div' }: GlassBarProps) {
  return (
    <Tag className={cn('glass sticky top-0 z-50', className)}>
      {children}
    </Tag>
  );
}
