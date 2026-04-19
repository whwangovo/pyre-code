// Shared theme + nav rendering
(function () {
  const stored = localStorage.getItem('pyre-theme');
  if (stored) document.body.setAttribute('data-theme', stored);

  window.setPyreTheme = function (t) {
    document.body.setAttribute('data-theme', t);
    localStorage.setItem('pyre-theme', t);
  };

  window.renderPyreNav = function (opts) {
    opts = opts || {};
    const active = opts.active || '';
    const pages = [
      { key: 'home',     href: 'Home.html',          label: 'Home' },
      { key: 'problems', href: 'Problems.html',      label: 'Problems' },
      { key: 'paths',    href: 'Paths.html',         label: 'Paths' },
    ];
    const nav = document.createElement('nav');
    nav.className = 'top';
    nav.innerHTML = `
      <div class="nav-inner">
        <div style="display:flex;align-items:center;gap:32px;">
          <a href="Home.html" class="brand">
            <span class="glyph" aria-hidden="true">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M6 1.25c.6 1.8.15 2.7-.75 3.6C4 6.2 3.25 7.2 3.25 8.5a2.75 2.75 0 1 0 5.5 0c0-1-.3-1.8-1-2.6.3 1.1-.15 1.8-.8 1.8-.5 0-.85-.4-.85-1C6.1 5.6 6.7 3.9 6 1.25Z" fill="currentColor"/></svg>
            </span>
            Pyre Code
          </a>
          <div class="nav-links">
            ${pages.map(p => `<a href="${p.href}" class="${active===p.key?'active':''}">${p.label}</a>`).join('')}
          </div>
        </div>
        <div class="nav-right">
          <div class="chip"><span class="dot"></span><span>12 / 68 solved</span></div>
          <button class="icon-btn" id="pyreThemeBtn" title="Toggle theme">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
          </button>
          <button class="pill-btn">EN</button>
        </div>
      </div>`;
    document.body.prepend(nav);
    nav.querySelector('#pyreThemeBtn').addEventListener('click', () => {
      const cur = document.body.getAttribute('data-theme') || 'light';
      setPyreTheme(cur === 'light' ? 'dark' : 'light');
    });
  };

  window.renderPyreFooter = function () {
    const f = document.createElement('footer');
    f.className = 'site';
    f.innerHTML = `
      <div class="foot-inner">
        <div>PYRE_CODE · v0.4.2 · MIT · Built on torch_judge</div>
        <div style="display:flex;gap:18px;">
          <a href="#">GitHub</a>
          <a href="#">Issues</a>
          <a href="#">Contribute</a>
          <a href="#">EN / 中文</a>
        </div>
      </div>`;
    document.body.appendChild(f);
  };
})();
