'use client';

import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';

type Design = 'classic' | 'redesign';

interface DesignContextValue {
  design: Design;
  setDesign: (d: Design) => void;
  toggleDesign: () => void;
}

const DesignContext = createContext<DesignContextValue>({
  design: 'redesign',
  setDesign: () => {},
  toggleDesign: () => {},
});

export function DesignProvider({ children }: { children: ReactNode }) {
  const [design, setDesignState] = useState<Design>('redesign');

  useEffect(() => {
    const stored = localStorage.getItem('pyre-design') as Design | null;
    if (stored === 'classic' || stored === 'redesign') {
      setDesignState(stored);
    }
  }, []);

  const setDesign = (d: Design) => {
    setDesignState(d);
    localStorage.setItem('pyre-design', d);
    document.documentElement.setAttribute('data-design', d);
  };

  const toggleDesign = () => {
    setDesign(design === 'redesign' ? 'classic' : 'redesign');
  };

  useEffect(() => {
    document.documentElement.setAttribute('data-design', design);
  }, [design]);

  return (
    <DesignContext.Provider value={{ design, setDesign, toggleDesign }}>
      {children}
    </DesignContext.Provider>
  );
}

export function useDesign() {
  return useContext(DesignContext);
}
