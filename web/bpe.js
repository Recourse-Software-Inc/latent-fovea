// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// GPT-2 byte-level BPE tokenizer (encode + decode) from vocab.json and merges.txt.
// Same algorithm as OpenAI's encoder.py; works in browsers and Node.

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const map = new Map();
  bs.forEach((b, i) => map.set(b, String.fromCharCode(cs[i])));
  return map;
}

const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class GPT2Tokenizer {
  constructor(vocab, mergesText) {
    this.encoder = vocab;
    this.decoder = new Map(Object.entries(vocab).map(([k, v]) => [v, k]));
    this.byteEnc = bytesToUnicode();
    this.byteDec = new Map([...this.byteEnc].map(([b, c]) => [c, b]));
    const lines = mergesText.split('\n').filter((l) => l && !l.startsWith('#version'));
    this.ranks = new Map(lines.map((l, i) => [l, i]));
    this.cache = new Map();
    this.utf8 = new TextEncoder();
    this.utf8d = new TextDecoder('utf-8', { fatal: false });
  }

  bpe(token) {
    if (this.cache.has(token)) return this.cache.get(token);
    let word = [...token];
    while (word.length > 1) {
      let best = null, bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(word[i] + ' ' + word[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best === null) break;
      const a = word[best], b = word[best + 1], next = [];
      for (let i = 0; i < word.length;) {
        if (i < word.length - 1 && word[i] === a && word[i + 1] === b) { next.push(a + b); i += 2; }
        else { next.push(word[i]); i++; }
      }
      word = next;
    }
    this.cache.set(token, word);
    return word;
  }

  encode(text) {
    const ids = [];
    for (const m of text.matchAll(PAT)) {
      const piece = [...this.utf8.encode(m[0])].map((b) => this.byteEnc.get(b)).join('');
      for (const t of this.bpe(piece)) ids.push(this.encoder[t]);
    }
    return ids;
  }

  decode(ids) {
    const s = ids.map((i) => this.decoder.get(i) || '').join('');
    return this.utf8d.decode(new Uint8Array([...s].map((c) => this.byteDec.get(c))));
  }
}

// ---- Hugging Face tokenizer.json (byte-level BPE: GPT-2, Llama 3, Qwen, SmolLM, Pythia) ----

// Hugging Face regexes use (?i:...) groups, which not every browser supports: spell the
// letters inside such groups out as [xX] classes instead.
function jsRegex(src) {
  let out = '', i = 0;
  while (i < src.length) {
    if (src.startsWith('(?i:', i)) {
      let depth = 1, j = i + 4;
      for (; j < src.length && depth; j++) { if (src[j] === '\\') { j++; continue; } if (src[j] === '(') depth++; else if (src[j] === ')') depth--; }
      const inner = src.slice(i + 4, j - 1).replace(/[a-zA-Z]/g, (c) => `[${c.toLowerCase()}${c.toUpperCase()}]`);
      out += `(?:${inner})`; i = j;
    } else { out += src[i]; i++; }
  }
  return new RegExp(out, 'gu');
}

// Split `pieces` by a regex, keeping matches and the text between them as pieces.
function splitIsolated(pieces, re) {
  const out = [];
  for (const p of pieces) {
    let last = 0;
    for (const m of p.matchAll(re)) {
      if (m.index > last) out.push(p.slice(last, m.index));
      if (m[0]) out.push(m[0]);
      last = m.index + m[0].length;
    }
    if (last < p.length) out.push(p.slice(last));
  }
  return out;
}

export class HFTokenizer {
  constructor(tj) {
    const model = tj.model;
    this.encoder = new Map(Object.entries(model.vocab));
    for (const a of tj.added_tokens || []) this.encoder.set(a.content, a.id);
    this.decoder = new Map([...this.encoder].map(([k, v]) => [v, k]));
    this.ranks = new Map(model.merges.map((m, i) => [Array.isArray(m) ? m.join(' ') : m, i]));
    this.ignoreMerges = !!model.ignore_merges;
    this.byteEnc = bytesToUnicode();
    this.byteDec = new Map([...this.byteEnc].map(([b, c]) => [c, b]));
    this.utf8 = new TextEncoder(); this.utf8d = new TextDecoder('utf-8', { fatal: false });
    this.cache = new Map();
    this.nfc = JSON.stringify(tj.normalizer || {}).includes('"NFC"');
    this.steps = [];
    const walk = (pt) => {
      if (!pt) return;
      if (pt.type === 'Sequence') return pt.pretokenizers.forEach(walk);
      if (pt.type === 'ByteLevel') {
        if (pt.add_prefix_space) this.steps.push((ps) => ps.map((p, i) => (i === 0 && !p.startsWith(' ') ? ' ' + p : p)));
        if (pt.use_regex !== false) { const re = new RegExp(PAT.source, 'gu'); this.steps.push((ps) => splitIsolated(ps, re)); }
      } else if (pt.type === 'Split') {
        const re = pt.pattern.Regex ? jsRegex(pt.pattern.Regex) : new RegExp(pt.pattern.String.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gu');
        if (pt.invert || (pt.behavior && pt.behavior !== 'Isolated')) throw new Error(`pre-tokenizer split behaviour ${pt.behavior} is not supported`);
        this.steps.push((ps) => splitIsolated(ps, re));
      } else if (pt.type === 'Digits') {
        const re = pt.individual_digits ? /\p{Nd}/gu : /\p{Nd}+/gu;
        this.steps.push((ps) => splitIsolated(ps, re));
      } else throw new Error(`pre-tokenizer ${pt.type} is not supported`);
    };
    walk(tj.pre_tokenizer);
    // A leading special token from the post-processor (Llama 3's <|begin_of_text|>).
    const single = tj.post_processor?.single || tj.post_processor?.processors?.find((p) => p.single)?.single;
    const first = single && single[0] && single[0].SpecialToken;
    this.bos = first ? this.encoder.get(first.id) : undefined;
    const specials = (tj.added_tokens || []).map((a) => a.content).sort((a, b) => b.length - a.length);
    this.specialRe = specials.length ? new RegExp(specials.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g') : null;
  }

  bpe(token) {
    if (this.cache.has(token)) return this.cache.get(token);
    if (this.ignoreMerges && this.encoder.has(token)) { this.cache.set(token, [token]); return [token]; }
    let word = [...token];
    while (word.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(word[i] + ' ' + word[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = word[best], b = word[best + 1], next = [];
      for (let i = 0; i < word.length;) {
        if (i < word.length - 1 && word[i] === a && word[i + 1] === b) { next.push(a + b); i += 2; } else { next.push(word[i]); i++; }
      }
      word = next;
    }
    this.cache.set(token, word);
    return word;
  }

  encode(text) {
    if (this.nfc) text = text.normalize('NFC');
    const ids = this.bos !== undefined ? [this.bos] : [];
    const chunks = [];
    if (this.specialRe) {  // special tokens typed literally stay whole
      let last = 0;
      for (const m of text.matchAll(this.specialRe)) { if (m.index > last) chunks.push([text.slice(last, m.index), false]); chunks.push([m[0], true]); last = m.index + m[0].length; }
      if (last < text.length) chunks.push([text.slice(last), false]);
    } else chunks.push([text, false]);
    for (const [chunk, special] of chunks) {
      if (special) { ids.push(this.encoder.get(chunk)); continue; }
      let pieces = [chunk];
      for (const f of this.steps) pieces = f(pieces);
      for (const p of pieces) {
        const bytes = [...this.utf8.encode(p)].map((b) => this.byteEnc.get(b)).join('');
        for (const t of this.bpe(bytes)) { const id = this.encoder.get(t); if (id !== undefined) ids.push(id); }
      }
    }
    return ids;
  }

  decode(ids) {
    const s = ids.map((i) => this.decoder.get(i) || '').join('');
    return this.utf8d.decode(new Uint8Array([...s].map((c) => this.byteDec.get(c)).filter((b) => b !== undefined)));
  }
}
