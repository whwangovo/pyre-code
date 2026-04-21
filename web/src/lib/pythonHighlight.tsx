import { ReactNode } from 'react';

const COLORS = {
  keyword:  'var(--code-keyword)',
  string:   'var(--code-string)',
  comment:  'var(--code-comment)',
  number:   'var(--code-number)',
  builtin:  'var(--code-builtin)',
  funcname: 'var(--code-func)',
  default:  'var(--text)',
};

const KEYWORDS = new Set([
  'import','from','def','class','return','if','else','elif','for','in',
  'while','try','except','with','as','pass','raise','assert','not','and',
  'or','is','None','True','False','lambda','del','global','nonlocal',
  'yield','break','continue','finally',
]);

const BUILTINS = new Set([
  'torch','print','len','range','isinstance','type','int','float','str',
  'list','dict','tuple','set','bool','abs','max','min','sum','zip','map',
  'enumerate','hasattr','getattr','setattr','super','self','F','nn',
]);

type Token = { text: string; color: string };

function tokenize(code: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < code.length) {
    // Comment
    if (code[i] === '#') {
      const end = code.indexOf('\n', i);
      const text = end === -1 ? code.slice(i) : code.slice(i, end);
      tokens.push({ text, color: COLORS.comment });
      i += text.length;
      continue;
    }

    // Triple-quoted string
    if (code.startsWith('"""', i) || code.startsWith("'''", i)) {
      const q = code.slice(i, i + 3);
      const end = code.indexOf(q, i + 3);
      const text = end === -1 ? code.slice(i) : code.slice(i, end + 3);
      tokens.push({ text, color: COLORS.string });
      i += text.length;
      continue;
    }

    // Single-quoted string
    if (code[i] === '"' || code[i] === "'") {
      const q = code[i];
      let j = i + 1;
      while (j < code.length && code[j] !== q && code[j] !== '\n') {
        if (code[j] === '\\') j++;
        j++;
      }
      const text = code.slice(i, j + 1);
      tokens.push({ text, color: COLORS.string });
      i = j + 1;
      continue;
    }

    // Number
    if (/[0-9]/.test(code[i]) || (code[i] === '-' && /[0-9]/.test(code[i + 1] || ''))) {
      let j = i;
      if (code[j] === '-') j++;
      while (j < code.length && /[0-9._eE+\-]/.test(code[j])) j++;
      tokens.push({ text: code.slice(i, j), color: COLORS.number });
      i = j;
      continue;
    }

    // Word (keyword, builtin, identifier, function call)
    if (/[a-zA-Z_]/.test(code[i])) {
      let j = i;
      while (j < code.length && /[a-zA-Z0-9_]/.test(code[j])) j++;
      const word = code.slice(i, j);
      const isFuncCall = code[j] === '(';
      let color = COLORS.default;
      if (KEYWORDS.has(word)) color = COLORS.keyword;
      else if (BUILTINS.has(word)) color = COLORS.builtin;
      else if (isFuncCall) color = COLORS.funcname;
      tokens.push({ text: word, color });
      i = j;
      continue;
    }

    // Default: single char
    tokens.push({ text: code[i], color: COLORS.default });
    i++;
  }

  return tokens;
}

export function PythonCode({ code }: { code: string }): ReactNode {
  const tokens = tokenize(code);
  return (
    <>
      {tokens.map((tok, i) => (
        <span key={i} style={{ color: tok.color }}>
          {tok.text}
        </span>
      ))}
    </>
  );
}
