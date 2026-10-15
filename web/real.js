// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// The handbook page (real.html): its navigation, and closing it when shown over the viewer.
// Shown in a window over the viewer (?embed): the back link closes the window instead.
if (new URLSearchParams(location.search).has('embed') && parent !== window) {
  document.documentElement.classList.add('embed');
  const b = document.getElementById('back');
  b.textContent = 'Back to the viewer';
  b.addEventListener('click', (e) => { e.preventDefault(); parent.postMessage('lf-close-explain', location.origin); });
  addEventListener('keydown', (e) => { if (e.key === 'Escape') parent.postMessage('lf-close-explain', location.origin); });
}

// ---------- navigation ----------
(() => {
  const side = document.getElementById('side'), tocBtn = document.getElementById('tocBtn'), topBtn = document.getElementById('topBtn');
  const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const title = (el) => (el.querySelector('h2') || {}).textContent || '';
  const targets = [];  // in page order: { id, el }
  let html = '<div class="side-h">Is it real?</div><a href="#top" data-id="top">Summary of this tool</a>';
  targets.push({ id: 'top', el: document.getElementById('top') });
  for (const sec of document.querySelectorAll('.wrap > section[data-nav]')) {
    html += `<a href="#${sec.id}" data-id="${sec.id}">${esc(sec.dataset.nav)}</a>`;
    targets.push({ id: sec.id, el: sec });
  }
  html += '<div class="side-h">The handbook</div><a href="#handbook" data-id="handbook">Contents</a>';
  targets.push({ id: 'handbook', el: document.getElementById('handbook') });
  const chapters = [...document.querySelectorAll('article.ch')];
  for (const part of document.querySelectorAll('section.part')) {
    const kick = (part.querySelector(':scope > .kicker') || {}).textContent || '';
    html += `<div class="side-h">${esc(kick.replace(/^Part [IV]+ · /, (m) => m))}</div>`;
    for (const ch of part.querySelectorAll('article.ch')) {
      const k = (ch.querySelector('.kicker') || {}).textContent || '';
      const n = (k.match(/(\d+|[A-Z])$/) || [''])[0];
      html += `<a href="#${ch.id}" data-id="${ch.id}"><span class="n">${n}</span>${esc(title(ch))}</a>`;
      targets.push({ id: ch.id, el: ch });
    }
  }
  side.innerHTML = html;

  // Anchors on chapter and section headings; subsections get ids from their chapter.
  for (const t of targets) {
    const h = t.el.tagName === 'H1' ? null : t.el.querySelector('h2');
    if (h && !h.querySelector('.anchor')) h.insertAdjacentHTML('beforeend', `<a class="anchor" href="#${t.id}" aria-label="Link to this section">#</a>`);
  }
  for (const ch of chapters) ch.querySelectorAll('h3').forEach((h, i) => {
    if (!h.id) h.id = `${ch.id}-${i + 1}`;
    h.insertAdjacentHTML('beforeend', `<a class="anchor" href="#${h.id}" aria-label="Link to this subsection">#</a>`);
  });

  // Previous and next chapter.
  chapters.forEach((ch, i) => {
    const p = chapters[i - 1], q = chapters[i + 1], lab = (c) => `${esc((c.querySelector('.kicker') || {}).textContent || '')}`;
    let nav = '<nav class="pn" aria-label="Chapter navigation">';
    if (p) nav += `<a class="prev" href="#${p.id}"><small>← ${lab(p)}</small>${esc(title(p))}</a>`;
    if (q) nav += `<a class="next" href="#${q.id}"><small>${lab(q)} →</small>${esc(title(q))}</a>`;
    else nav += '<a class="next" href="#handbook"><small>↑ Back</small>Contents</a>';
    ch.insertAdjacentHTML('beforeend', nav + '</nav>');
  });

  // Where am I? The last target whose top has passed the upper part of the window.
  const links = new Map([...side.querySelectorAll('a[data-id]')].map((a) => [a.dataset.id, a]));
  // Keep an entry visible inside the list itself (never scrolling the page).
  const reveal = (a, centre) => {
    if (!a || side.clientHeight === 0) return;
    const top = a.offsetTop, h = a.offsetHeight, view = side.clientHeight;
    if (centre) side.scrollTop = top - view / 2 + h / 2;
    else if (top < side.scrollTop + 20) side.scrollTop = top - 20;
    else if (top + h > side.scrollTop + view - 20) side.scrollTop = top + h - view + 20;
  };
  let current = '';
  const spy = () => {
    let id = targets[0].id;
    for (const t of targets) if (t.el && t.el.getBoundingClientRect().top < innerHeight * 0.3) id = t.id;
    if (id !== current) {
      links.get(current)?.classList.remove('on');
      const a = links.get(id); a?.classList.add('on'); current = id;
      reveal(a, false);
    }
    topBtn.hidden = scrollY < 900;
  };
  let raf = 0;
  addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; spy(); }); }, { passive: true });
  spy();

  // The contents button opens the list as a sheet; choosing an entry, Escape or a click outside closes it.
  const close = () => { side.classList.remove('sheet'); tocBtn.setAttribute('aria-expanded', 'false'); };
  tocBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = !side.classList.contains('sheet');
    side.classList.toggle('sheet', open); tocBtn.setAttribute('aria-expanded', String(open));
    if (open) reveal(links.get(current), true);
  });
  side.addEventListener('click', (e) => { if (e.target.closest('a') && side.classList.contains('sheet')) close(); });
  document.addEventListener('click', (e) => { if (side.classList.contains('sheet') && !side.contains(e.target)) close(); });
  addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  topBtn.addEventListener('click', () => scrollTo({ top: 0, behavior: 'smooth' }));

  // Glossary filter.
  const gl = document.querySelector('.gloss');
  if (gl) {
    gl.insertAdjacentHTML('beforebegin', '<input type="search" class="gloss-find" id="glossFind" placeholder="Find a term…" aria-label="Find a term in the glossary" autocomplete="off" spellcheck="false" />');
    document.getElementById('glossFind').addEventListener('input', (e) => {
      const q = e.target.value.trim().toLowerCase();
      for (const tr of gl.rows) tr.classList.toggle('hide', !!q && !tr.textContent.toLowerCase().includes(q));
    });
  }
})();
