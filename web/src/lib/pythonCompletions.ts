/**
 * Python completions for the Monaco editor.
 *
 * Monaco ships a Python *tokenizer*, not a language service, so out of the box
 * the only suggestions are words that already appear in the buffer -- typing
 * `np.` or `torch.` offers nothing. This registers a completion provider backed
 * by `pythonCompletions.json`, which `scripts/gen_completions.py` generates by
 * introspecting the exact modules the grading sandbox injects into every
 * submission (see `grading_service/main.py`): torch, nn, F, np, math, Tensor.
 *
 * The data is ~300 KB, so it is imported dynamically: it lands in the same lazy
 * chunk as the editor itself and never reaches the initial page bundle.
 */

import type * as Monaco from 'monaco-editor';

type Kind = 'function' | 'class' | 'module' | 'constant' | 'property';

interface Entry {
  /** label */
  l: string;
  /** kind */
  k: Kind;
  /** detail: signature for callables, value for constants */
  d?: string;
  /** summary: first line of the docstring */
  s?: string;
}

interface CompletionData {
  versions: Record<string, string>;
  keywords: string[];
  builtins: string[];
  namespaces: Record<string, Entry[]>;
}

/**
 * Dotted prefixes that resolve to a generated namespace. The sandbox binds the
 * short aliases, but people type the long form out of habit.
 */
const ALIASES: Record<string, string> = {
  torch: 'torch',
  nn: 'nn',
  'torch.nn': 'nn',
  F: 'F',
  'nn.functional': 'F',
  'torch.nn.functional': 'F',
  np: 'np',
  numpy: 'np',
  math: 'math',
  Tensor: 'Tensor',
  'torch.Tensor': 'Tensor',
};

/** `foo.bar` / `foo.` — captures the dotted receiver left of the final dot. */
const MEMBER_ACCESS = /([A-Za-z_][A-Za-z0-9_.]*)\.\s*[A-Za-z0-9_]*$/;

let registration: Monaco.IDisposable | null = null;

function kindOf(monaco: typeof Monaco, kind: Kind): Monaco.languages.CompletionItemKind {
  const K = monaco.languages.CompletionItemKind;
  switch (kind) {
    case 'function':
      return K.Function;
    case 'class':
      return K.Class;
    case 'module':
      return K.Module;
    case 'property':
      return K.Property;
    default:
      return K.Constant;
  }
}

/**
 * True when the cursor sits inside a string or a comment, where API
 * suggestions are noise. Deliberately cheap: it only inspects the current line,
 * so it misses triple-quoted blocks spanning lines.
 */
function inStringOrComment(linePrefix: string): boolean {
  let single = 0;
  let double = 0;
  for (let i = 0; i < linePrefix.length; i++) {
    const c = linePrefix[i];
    if (c === '\\') {
      i++;
    } else if (c === "'" && double % 2 === 0) {
      single++;
    } else if (c === '"' && single % 2 === 0) {
      double++;
    } else if (c === '#' && single % 2 === 0 && double % 2 === 0) {
      return true;
    }
  }
  return single % 2 === 1 || double % 2 === 1;
}

export async function registerPythonCompletions(monaco: typeof Monaco): Promise<void> {
  if (registration) return; // every editor instance shares one provider
  registration = { dispose: () => {} }; // claim the slot before the await

  const data = ((await import('./pythonCompletions.json')) as unknown as { default: CompletionData })
    .default;

  const build = (
    entries: Entry[],
    range: Monaco.IRange
  ): Monaco.languages.CompletionItem[] =>
    entries.map((e) => ({
      label: e.l,
      kind: kindOf(monaco, e.k),
      detail: e.d,
      documentation: e.s ? { value: e.s } : undefined,
      insertText: e.l,
      range,
    }));

  registration = monaco.languages.registerCompletionItemProvider('python', {
    triggerCharacters: ['.'],

    provideCompletionItems(model, position) {
      const linePrefix = model.getValueInRange({
        startLineNumber: position.lineNumber,
        startColumn: 1,
        endLineNumber: position.lineNumber,
        endColumn: position.column,
      });

      if (inStringOrComment(linePrefix)) return { suggestions: [] };

      const word = model.getWordUntilPosition(position);
      const range: Monaco.IRange = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };

      const member = MEMBER_ACCESS.exec(linePrefix);
      if (member) {
        const receiver = member[1];
        const namespace =
          ALIASES[receiver] ?? ALIASES[receiver.split('.').pop() ?? ''] ?? null;

        if (namespace) {
          return { suggestions: build(data.namespaces[namespace] ?? [], range) };
        }

        // Unknown receiver. In these problems essentially every local is a
        // tensor, so Tensor members are the useful guess -- and far better than
        // the editor's only alternative, which is other words in the file.
        return { suggestions: build(data.namespaces.Tensor ?? [], range) };
      }

      // Top level: the sandbox globals, then keywords and builtins.
      const suggestions: Monaco.languages.CompletionItem[] = Object.keys(data.namespaces).map(
        (alias) => ({
          label: alias,
          kind: monaco.languages.CompletionItemKind.Module,
          detail: alias === 'F' ? 'torch.nn.functional' : undefined,
          insertText: alias,
          range,
        })
      );

      for (const kw of data.keywords) {
        suggestions.push({
          label: kw,
          kind: monaco.languages.CompletionItemKind.Keyword,
          insertText: kw,
          range,
        });
      }
      for (const fn of data.builtins) {
        suggestions.push({
          label: fn,
          kind: monaco.languages.CompletionItemKind.Function,
          detail: 'builtin',
          insertText: fn,
          range,
        });
      }

      return { suggestions };
    },
  });
}
