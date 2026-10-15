// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Writes one file to the browser's storage for the main page. Safari only allows the
// sync access handles store.js falls back to inside a worker. In: { path, data }.  Out: { ok, message? }.
import { writeFile } from './store.js';

self.onmessage = async ({ data: m }) => {
  try { await writeFile(m.path, m.data); self.postMessage({ ok: true }); }
  catch (e) { self.postMessage({ ok: false, message: e.message }); }
};
