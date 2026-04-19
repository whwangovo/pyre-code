'use client';

import { useLocale } from '@/context/LocaleContext';

export function Footer() {
  const { t } = useLocale();

  return (
    <footer
      className="mt-4 text-[12.5px] mono text-text-3"
      style={{
        borderTop: '1px solid var(--line)',
        padding: '28px 0 44px',
      }}
    >
      <div className="max-w-[1280px] mx-auto px-7 flex justify-between gap-4 flex-wrap">
        <div>{t('footerBrand')}</div>
        <div className="flex gap-[18px]">
          <a href="https://github.com/whwangovo/pyre-code" className="hover:text-text transition-colors duration-150">GitHub</a>
          <a href="https://github.com/whwangovo/pyre-code/issues" className="hover:text-text transition-colors duration-150">Issues</a>
        </div>
      </div>
    </footer>
  );
}
