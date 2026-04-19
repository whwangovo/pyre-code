'use client';

import { useRef, useEffect } from 'react';
import Editor, { type OnMount } from '@monaco-editor/react';
import { appleLight, appleDark } from '@/lib/monacoTheme';
import { useTheme } from '@/context/ThemeContext';

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  height?: string;
  allowParentScrollOnWheel?: boolean;
}

export function CodeEditor({
  value,
  onChange,
  readOnly = false,
  height = '100%',
  allowParentScrollOnWheel = false,
}: CodeEditorProps) {
  const editorRef = useRef<any>(null);
  const monacoRef = useRef<any>(null);
  const { theme } = useTheme();

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    monaco.editor.defineTheme('apple-light', appleLight);
    monaco.editor.defineTheme('apple-dark', appleDark);
    monaco.editor.setTheme(theme === 'dark' ? 'apple-dark' : 'apple-light');
  };

  useEffect(() => {
    if (monacoRef.current) {
      monacoRef.current.editor.setTheme(theme === 'dark' ? 'apple-dark' : 'apple-light');
    }
  }, [theme]);

  return (
    <Editor
      height={height}
      language="python"
      value={value}
      onChange={(v) => onChange(v || '')}
      onMount={handleMount}
      options={{
        readOnly,
        fontSize: 14,
        lineHeight: 22,
        fontFamily: "'Geist Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        padding: { top: 16, bottom: 16 },
        renderLineHighlight: 'line',
        smoothScrolling: true,
        cursorBlinking: 'smooth',
        cursorSmoothCaretAnimation: 'on',
        bracketPairColorization: { enabled: true },
        overviewRulerBorder: false,
        hideCursorInOverviewRuler: true,
        scrollbar: {
          alwaysConsumeMouseWheel: !allowParentScrollOnWheel,
          verticalScrollbarSize: 6,
          horizontalScrollbarSize: 6,
        },
      }}
    />
  );
}
