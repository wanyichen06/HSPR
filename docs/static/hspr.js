(() => {
  const body = document.body;
  const toggle = document.getElementById('language-toggle');
  function setLanguage(language) {
    const chosen = language === 'en' ? 'en' : 'zh';
    body.classList.remove('lang-en', 'lang-zh');
    body.classList.add('lang-' + chosen);
    document.documentElement.lang = chosen;
    toggle.setAttribute('aria-label', chosen === 'zh' ? 'Switch to English' : '切换为中文');
    try { localStorage.setItem('hspr-language', chosen); } catch (_) { /* Storage is optional. */ }
  }
  const requested = new URLSearchParams(location.search).get('lang');
  let saved = null;
  try { saved = localStorage.getItem('hspr-language'); } catch (_) { /* Default to Chinese. */ }
  setLanguage(requested || saved || 'zh');
  toggle.addEventListener('click', () => {
    const next = body.classList.contains('lang-zh') ? 'en' : 'zh';
    setLanguage(next);
    const address = new URL(location.href);
    address.searchParams.set('lang', next);
    history.replaceState(null, '', address);
  });
  document.getElementById('copy-citation').addEventListener('click', async () => {
    const status = document.getElementById('copy-status');
    const chinese = body.classList.contains('lang-zh');
    const citation = document.getElementById('bibtex');
    try {
      await navigator.clipboard.writeText(citation.textContent);
      status.textContent = chinese ? 'BibTeX 已复制。' : 'BibTeX copied.';
    } catch (_) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(citation);
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = chinese ? '已选中引用，请手动复制。' : 'Citation selected; copy it manually.';
    }
  });
})();
