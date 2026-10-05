// Humanizer core: runs fully on-device. No network after the models are cached.
//
// Method (see RESEARCH.md): build a statistical profile of what zero-shot detectors look at
// (GLTR token ranks, Fast-DetectGPT discrepancy, burstiness of surprisal, sentence lengths,
// punctuation), generate several variants of every sentence, score each variant once with a
// small local LM, then pick the combination whose assembled profile is closest to a target
// human profile (the user's own writing, or a typicality target when no sample is given).

import { env, AutoTokenizer, AutoModelForCausalLM, pipeline, Tensor } from "./vendor/transformers.min.js";

// ---------------------------------------------------------------- config
export const MODELS = {
  lm: "onnx-community/Qwen2.5-0.5B-Instruct",
  embed: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
};
const CTX_TOKENS = 48;           // left context used when scoring a sentence
const TOPK = 8;                  // alternatives remembered per position for word swaps

export function configure({ local = false } = {}) {
  env.backends.onnx.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
  env.useBrowserCache = true;
  if (local) {               // self-test with models served next to the app
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = new URL("./models/", import.meta.url).pathname;
    env.useBrowserCache = false;
    MODELS.lm = "test-lm";
    MODELS.embed = "test-embed";
  }
}

// ---------------------------------------------------------------- text utils
const WORD_RE = /\p{L}+(?:-\p{L}+)*/gu;
export const words = t => t.match(WORD_RE) || [];
export const detectLang = t => ((t.match(/[а-яё]/gi) || []).length >= (t.match(/[a-z]/gi) || []).length ? "ru" : "en");
export function splitSentences(t) {
  const out = [];
  for (const para of t.split(/\n+/)) {
    for (const s of para.split(/(?<=[.!?…])["»)]?\s+(?=["«(]?[A-ZА-ЯЁ0-9])/u)) if (s.trim()) out.push(s.trim());
  }
  return out;
}
export function makeDoc(text) {
  const sents = splitSentences(text), seps = [];
  let pos = 0;
  for (const s of sents) { const i = text.indexOf(s, pos); seps.push(text.slice(pos, i)); pos = i + s.length; }
  seps.push(text.slice(pos));
  return { sents, render: arr => seps.map((sp, i) => sp + (i < arr.length ? arr[i] : "")).join("") };
}
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
export const mean = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);
export const cv = a => { if (a.length < 2) return 0; const m = mean(a); return m ? Math.sqrt(mean(a.map(x => (x - m) ** 2))) / m : 0; };
export function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const scriptOf = w => /[а-яё]/i.test(w) ? "cyr" : /[a-z]/i.test(w) ? "lat" : "other";
// share of letters written in the expected script (names and abbreviations aside, should be ~1)
const scriptShare = (t, lang) => { const all = (t.match(/\p{L}/gu) || []).length; const ok = (t.match(lang === "ru" ? /[а-яё]/gi : /[a-z]/gi) || []).length; return all ? ok / all : 0; };

// ---------------------------------------------------------------- clichés (Kobak et al. style excess vocabulary + connectors)
export const CLICHES = {
  ru: [
    [/Важно отметить, что /g, [""]], [/Стоит отметить, что /g, ["", "Кстати, "]], [/Следует отметить, что /g, [""]],
    [/Необходимо подчеркнуть, что /g, [""]], [/Кроме того, /g, ["А ещё ", "К тому же ", ""]],
    [/Более того, /g, ["Да и ", ""]], [/Помимо этого, /g, ["Ещё ", ""]],
    [/Таким образом, /g, ["Выходит, ", "Итак, ", "В итоге ", ""]], [/В заключение,? /g, ["Если коротко, ", "В итоге ", ""]],
    [/В целом, /g, ["В общем, ", ""]], [/[Вв] современном мире/g, ["сегодня", "сейчас"]],
    [/играет ключевую роль/g, ["очень важен", "много значит"]], [/играет важную роль/g, ["важен", "заметно влияет"]],
    [/является неотъемлемой частью/g, ["давно стал частью", "входит в"]], [/широкий спектр/g, ["целый ряд", "множество"]],
    [/Однако, /g, ["Но ", "Однако "]],
  ],
  en: [
    [/It(?: is|'s) (?:important|worth) (?:to note|noting) that /g, [""]], [/Moreover, /g, ["Also, ", "On top of that, ", "And ", ""]],
    [/Furthermore, /g, ["Also, ", "Plus, ", ""]], [/Additionally, /g, ["Also, ", "And ", ""]],
    [/In conclusion, /g, ["So, ", "All in all, ", ""]], [/Overall, /g, ["All told, ", ""]], [/Ultimately, /g, ["In the end, ", ""]],
    [/[Ii]n today's (?:fast-paced |digital |modern )?world\b/g, ["today", "these days"]],
    [/\bdelve(s?) into\b/g, ["dig$1 into", "look$1 at"]], [/\butiliz(e|es|ed|ing)\b/g, ["us$1"]], [/\bleverag(e|es|ed|ing)\b/g, ["us$1"]],
    [/\bplays a (?:crucial|pivotal|vital|key) role\b/g, ["matters a lot", "is central"]],
    [/\ba (?:wide|broad|diverse) (?:range|array) of\b/g, ["many", "all sorts of"]], [/\bseamless(?:ly)?\b/g, ["smooth"]],
    [/\brobust\b/g, ["solid"]], [/\btapestry\b/g, ["mix"]], [/\bnavigate the complexities of\b/g, ["deal with"]],
  ],
};
const DETECT_EXTRA = {
  ru: [/является/g, /в рамках/g, /данн(?:ый|ая|ое|ые)/g, /позволяет/g, /значительно/g, /различн(?:ые|ых)/g, /необходимо/g],
  en: [/\bcrucial\b/g, /\bfoster\b/g, /\benhance\b/g, /\bin the realm of\b/g, /\bsignificantly\b/g, /\bvarious\b/g],
};
export function findCliches(t, lang) {
  const found = [];
  for (const [re] of CLICHES[lang]) for (const m of t.matchAll(new RegExp(re.source, "g"))) found.push(m[0].replace(/[,\s]+$/, ""));
  for (const re of DETECT_EXTRA[lang]) for (const m of t.matchAll(re)) found.push(m[0]);
  return found;
}
export function removeCliches(t, lang, r) {
  for (const [re, alts] of CLICHES[lang]) {
    t = t.replace(re, (...args) => {
      const m = args[0], off = args[args.length - 2], str = args[args.length - 1];
      let rep = alts[Math.floor(r() * alts.length)].replace(/\$1/g, typeof args[1] === "string" ? args[1] : "");
      if (rep === "") return "";
      const before = str.slice(0, off);
      const atStart = !before.trim() || /[.!?…\n]\s*$/.test(before);
      return /^\p{Lu}/u.test(m) && atStart ? cap(rep) : rep;
    });
  }
  return t.replace(/(^|[.!?…]\s+|\n)(\p{Ll})/gu, (_, a, b) => a + b.toUpperCase());
}
const SPLIT = { ru: [", но ", ", а ", ", и ", "; ", ", поэтому ", ", однако ", ", который ", ", которые "], en: [", but ", ", and ", "; ", ", so ", ", which "] };
function splitVariant(s, lang) {
  for (const sp of SPLIT[lang]) {
    const j = s.indexOf(sp);
    if (j > 0 && words(s.slice(0, j)).length >= 5 && words(s.slice(j + sp.length)).length >= 5) {
      const conj = sp.replace(/[,;\s]/g, "");
      if (/^котор|^which$/.test(conj)) continue;   // relative clauses don't split cleanly
      const rest = s.slice(j + sp.length);
      return s.slice(0, j).replace(/[,;]$/, "") + ". " + (conj && conj !== "и" && conj !== "and" ? cap(conj + " " + rest) : cap(rest));
    }
  }
  return null;
}

// ---------------------------------------------------------------- models
let tok = null, lm = null, embedder = null, device = "wasm";

async function pickDevice() {
  try { if (navigator.gpu && await navigator.gpu.requestAdapter()) return "webgpu"; } catch {}
  return "wasm";
}

export async function loadModels(onProgress) {
  device = await pickDevice();
  const files = {};
  const cb = p => {
    if (p.status === "progress" && p.total) { files[p.file] = [p.loaded, p.total]; }
    if (p.status === "done" && files[p.file]) files[p.file][0] = files[p.file][1];
    const tot = Object.values(files).reduce((a, [, t]) => a + t, 0), got = Object.values(files).reduce((a, [l]) => a + l, 0);
    onProgress?.({ loaded: got, total: tot, file: p.file });
  };
  const local = MODELS.lm === "test-lm";
  const dtype = local ? "fp32" : device === "webgpu" ? "q4f16" : "q8";   // 483 MB or 512 MB
  tok = await AutoTokenizer.from_pretrained(MODELS.lm, { progress_callback: cb });
  lm = await AutoModelForCausalLM.from_pretrained(MODELS.lm, { dtype, device, progress_callback: cb });
  try {
    embedder = await pipeline("feature-extraction", MODELS.embed, { dtype: local ? "fp32" : "q8", device: "wasm", progress_callback: cb });
  } catch (e) {
    console.warn("embedder unavailable", e);
    embedder = null;
  }
  return { device, embedder: !!embedder };
}
export const modelsReady = () => !!lm;

// ---------------------------------------------------------------- scoring
function toF32(t) {
  const d = t.data;
  if (d instanceof Float32Array) return d;
  if (d instanceof Uint16Array) {          // raw float16 bits
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) {
      const h = d[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
    }
    return out;
  }
  return Float32Array.from(d);
}

const encode = text => Array.from(tok.encode(text, { add_special_tokens: false }));
const decode = ids => tok.decode(ids, { skip_special_tokens: true });

// Score `ids` given `ctx` (token ids). Returns per-token logp, rank, mu, var, plus top-k alternatives.
async function scoreIds(ctx, ids) {
  const bos = tok.bos_token_id ?? tok.eos_token_id;
  let c = ctx.slice(-CTX_TOKENS);
  if (!c.length && bos != null) c = [bos];
  const seq = c.concat(ids);
  const ii = new Tensor("int64", BigInt64Array.from(seq.map(BigInt)), [1, seq.length]);
  const am = new Tensor("int64", BigInt64Array.from(seq.map(() => 1n)), [1, seq.length]);
  const out = await lm({ input_ids: ii, attention_mask: am });
  const L = out.logits.dims[1], V = out.logits.dims[2];
  const data = toF32(out.logits);
  const res = { lp: [], rank: [], mu: [], vr: [], top: [] };
  const start = c.length - 1;
  for (let k = 0; k < ids.length; k++) {
    const row = start + k;
    if (row < 0) { res.lp.push(NaN); res.rank.push(-1); res.mu.push(NaN); res.vr.push(NaN); res.top.push([]); continue; }
    const off = row * V;
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (data[off + v] > mx) mx = data[off + v];
    let se = 0;
    for (let v = 0; v < V; v++) se += Math.exp(data[off + v] - mx);
    const lse = mx + Math.log(se);
    const tgt = data[off + ids[k]] - lse;
    let rank = 0, m1 = 0, m2 = 0;
    const top = [];   // [logp, id] kept sorted descending, length TOPK
    for (let v = 0; v < V; v++) {
      const l = data[off + v] - lse;
      if (l > tgt) rank++;
      const p = Math.exp(l);
      m1 += p * l; m2 += p * l * l;
      if (top.length < TOPK || l > top[top.length - 1][0]) {
        top.push([l, v]); top.sort((a, b) => b[0] - a[0]); if (top.length > TOPK) top.pop();
      }
    }
    res.lp.push(tgt); res.rank.push(rank); res.mu.push(m1); res.vr.push(m2 - m1 * m1); res.top.push(top);
  }
  out.logits.dispose?.();
  return res;
}

export async function scoreSentence(ctxText, sentence) {
  const ctx = ctxText ? encode(ctxText + " ") : [];
  const ids = encode(sentence);
  const st = await scoreIds(ctx, ids);
  return { text: sentence, ids, ...st };
}

// ---------------------------------------------------------------- profile
const RANK_BINS = [[0, 0], [1, 2], [3, 9], [10, 99], [100, 1e12]];
const PUNCT = { comma: /,/, dash: /[—–]/, colon: /[:;]/, question: /[?!]/, paren: /[()]/ };

export function profileOf(scored, lang) {
  const p = { n: 0, rankHist: [0, 0, 0, 0, 0], surprisal: [], fdgNum: 0, fdgDen: 0, sentSurp: [], sentLen: [], punct: {}, cliches: 0, words: 0 };
  for (const k in PUNCT) p.punct[k] = 0;
  for (const s of scored) {
    const sur = [];
    for (let i = 0; i < s.lp.length; i++) {
      if (Number.isNaN(s.lp[i])) continue;
      const r = s.rank[i];
      for (let b = 0; b < RANK_BINS.length; b++) if (r >= RANK_BINS[b][0] && r <= RANK_BINS[b][1]) { p.rankHist[b]++; break; }
      sur.push(-s.lp[i]); p.fdgNum += s.lp[i] - s.mu[i]; p.fdgDen += s.vr[i]; p.n++;
    }
    p.surprisal.push(...sur);
    if (sur.length >= 3) p.sentSurp.push(mean(sur));
    const w = words(s.text).length;
    p.sentLen.push(w); p.words += w;
    for (const k in PUNCT) if (PUNCT[k].test(s.text)) p.punct[k]++;
    p.cliches += findCliches(s.text, lang).length;
  }
  if (p.n) p.rankHist = p.rankHist.map(c => c / p.n);
  for (const k in PUNCT) p.punct[k] /= scored.length || 1;
  p.fdg = p.fdgNum / Math.sqrt(Math.max(p.fdgDen, 1e-9));       // Fast-DetectGPT z for the whole text
  p.fdgPerTok = p.n ? p.fdg / Math.sqrt(p.n) : 0;
  p.burst = cv(p.sentSurp);
  p.lenCv = cv(p.sentLen);
  p.top1 = p.rankHist[0];
  p.top10 = p.rankHist[0] + p.rankHist[1] + p.rankHist[2];
  p.clPer100 = p.words ? p.cliches / p.words * 100 : 0;
  return p;
}

function w1(a, b) {
  if (!a.length || !b.length) return 0;
  a = [...a].sort((x, y) => x - y); b = [...b].sort((x, y) => x - y);
  let s = 0; const q = 48;
  for (let i = 0; i < q; i++) s += Math.abs(a[Math.min(a.length - 1, Math.floor(i / q * a.length))] - b[Math.min(b.length - 1, Math.floor(i / q * b.length))]);
  return s / q;
}
const sd = a => Math.sqrt(mean(a.map(x => (x - mean(a)) ** 2))) || 1;

// Distance to a full target profile (built from the user's writing).
export function distance(p, t) {
  let d = 2.0 * p.rankHist.reduce((a, x, i) => a + Math.abs(x - t.rankHist[i]), 0);
  d += 1.0 * w1(p.surprisal, t.surprisal) / sd(t.surprisal);
  d += 1.5 * Math.abs(p.fdgPerTok - t.fdgPerTok) / 0.1;
  d += 1.0 * Math.abs(p.burst - t.burst) / 0.1;
  d += 0.7 * w1(p.sentLen, t.sentLen) / (mean(t.sentLen) || 10);
  d += 0.5 * Object.keys(PUNCT).reduce((a, k) => a + Math.abs(p.punct[k] - t.punct[k]), 0);
  d += 0.5 * Math.abs(p.clPer100 - t.clPer100);
  return d;
}
// Without a writing sample: a typicality target. Human text sits near the model's own expected
// log-likelihood (Fast-DetectGPT ~ 0) and fluctuates more between sentences than LLM text.
export function typicalityDistance(p) {
  let d = 1.5 * Math.abs(p.fdgPerTok) / 0.1;
  d += 1.0 * Math.max(0, 0.3 - p.burst) / 0.1;
  d += 0.7 * Math.max(0, 0.45 - p.lenCv) / 0.15;
  d += 0.5 * p.clPer100;
  return d;
}

// ---------------------------------------------------------------- variants
const PARA_PROMPTS = {
  ru: [
    "Перепиши предложение другими словами, сохранив смысл и все факты. Сделай его проще и короче. Ответь только новым предложением.",
    "Перепиши предложение, полностью изменив порядок слов и построение фразы, но сохранив смысл и все факты. Ответь только новым предложением.",
    "Перескажи это предложение живым разговорным языком, как написал бы человек, сохранив смысл. Ответь только новым текстом.",
  ],
  en: [
    "Rewrite the sentence in different words, keeping its meaning and all facts. Make it simpler and shorter. Reply with the new sentence only.",
    "Rewrite the sentence with a completely different word order and structure, keeping its meaning and all facts. Reply with the new sentence only.",
    "Retell this sentence in plain, natural language the way a person would write it, keeping the meaning. Reply with the new text only.",
  ],
};

async function paraphrase(sentence, context, lang, k) {
  const out = [];
  for (let i = 0; i < k; i++) {
    const messages = [
      { role: "system", content: PARA_PROMPTS[lang][i % PARA_PROMPTS[lang].length] },
      { role: "user", content: (context ? (lang === "ru" ? "Контекст: " : "Context: ") + context + "\n" : "") + (lang === "ru" ? "Предложение: " : "Sentence: ") + sentence },
    ];
    let inputs;
    try {
      inputs = tok.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true });
    } catch { inputs = tok(messages.map(m => m.content).join("\n") + "\n"); }
    const n0 = inputs.input_ids.dims.at(-1);
    const maxNew = Math.min(160, Math.ceil(encode(sentence).length * 1.8) + 8);
    const gen = await lm.generate({ ...inputs, max_new_tokens: maxNew, do_sample: true, temperature: 0.8, top_p: 0.9 });
    const ids = Array.from(gen.data ?? gen.tolist?.()[0] ?? []).slice(n0).map(Number);
    let txt = decode(ids).split("\n").map(s => s.trim()).find(Boolean) || "";
    txt = txt.replace(/^["«“]|["»”]$/g, "").replace(/^(Новое предложение|Ответ|Sentence|New sentence)\s*:\s*/i, "").trim();
    const r = words(txt).length / Math.max(1, words(sentence).length);
    if (txt && r > 0.4 && r < 2.2 && scriptOf(txt) === scriptOf(sentence) && txt !== sentence) out.push(txt);
  }
  return out;
}

// Single-token word swaps suggested by the scoring pass itself (positions the LM found too predictable).
function wordSwaps(scored, maxSwaps = 2) {
  const pieces = scored.ids.map(id => decode([id]));
  const slots = [];
  for (let i = 0; i < scored.ids.length; i++) {
    const p = pieces[i];
    const nextOk = i + 1 >= pieces.length || /^[\s.,;:!?)»—–-]/.test(pieces[i + 1]);
    if (!/^\s\p{L}{4,}$/u.test(p) || !nextOk || scored.rank[i] !== 0 || Math.exp(scored.lp[i]) < 0.35) continue;
    const alts = scored.top[i].filter(([l, id]) => id !== scored.ids[i] && /^\s\p{L}{3,}$/u.test(decode([id])) && scriptOf(decode([id])) === scriptOf(p) && Math.exp(l) > 0.02)
      .map(([, id]) => decode([id])).filter(a => a.trim().toLowerCase().slice(0, 4) !== p.trim().toLowerCase().slice(0, 4));
    if (alts.length) slots.push({ i, p: Math.exp(scored.lp[i]), alts });
  }
  slots.sort((a, b) => b.p - a.p);
  const out = [];
  for (const s of slots.slice(0, maxSwaps)) {
    for (const alt of s.alts.slice(0, 2)) {
      const np = pieces.slice(); np[s.i] = alt;
      out.push(np.join("").trim());
    }
  }
  return out;
}

async function embed(texts) {
  const t = await embedder(texts, { pooling: "mean", normalize: true });
  const d = t.dims.at(-1), arr = t.data;
  return texts.map((_, i) => arr.slice(i * d, (i + 1) * d));
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

// ---------------------------------------------------------------- main
export async function styleProfile(sampleText, onStep) {
  const lang = detectLang(sampleText), sents = splitSentences(sampleText), scored = [];
  for (let i = 0; i < sents.length; i++) {
    onStep?.(`Изучаю ваш стиль: предложение ${i + 1} из ${sents.length}`);
    scored.push(await scoreSentence(sents[i - 1] || "", sents[i]));
  }
  return profileOf(scored, lang);
}

export async function humanize(text, { lang, target = null, paraphrases = 2, minSim = 0.82, seed = 1, onStep, signal } = {}) {
  lang = lang || detectLang(text);
  const doc = makeDoc(text), n = doc.sents.length, r = rng(seed);
  const check = () => { if (signal?.aborted) throw new DOMException("cancelled", "AbortError"); };

  // 1. score originals
  const orig = [];
  for (let i = 0; i < n; i++) { check(); onStep?.(`Замеряю исходный текст: ${i + 1} из ${n}`); orig.push(await scoreSentence(doc.sents[i - 1] || "", doc.sents[i])); }

  // 2. variants
  const cands = [];
  for (let i = 0; i < n; i++) {
    check();
    const s = doc.sents[i], set = new Set([s]);
    const add = v => { if (v && v.trim()) set.add(v.trim()); };
    add(removeCliches(s, lang, r));
    add(splitVariant(s, lang));
    for (const v of wordSwaps(orig[i])) add(v);
    if (paraphrases) { onStep?.(`Придумываю варианты: предложение ${i + 1} из ${n}`); for (const v of await paraphrase(s, doc.sents[i - 1] || "", lang, paraphrases)) add(v); }
    const base = scriptShare(s, lang);
    cands.push([...set].filter((c, j) => j === 0 || scriptShare(c, lang) >= Math.min(0.9, base - 0.05)));
  }

  // 3. meaning filter
  if (embedder) {
    onStep?.("Проверяю, что смысл не изменился");
    for (let i = 0; i < n; i++) {
      check();
      if (cands[i].length < 2) continue;
      const vecs = await embed(cands[i]);
      cands[i] = cands[i].filter((c, j) => j === 0 || dot(vecs[0], vecs[j]) >= minSim);
    }
  }

  // 4. score variants (context = previous original sentence, so every variant is scored exactly once)
  const scored = orig.map(o => [o]);
  const total = cands.reduce((a, c) => a + c.length - 1, 0);
  let done = 0;
  const meanSur = sc => mean(sc.lp.filter(x => !Number.isNaN(x)).map(x => -x));
  for (let i = 0; i < n; i++) {
    const keep = [cands[i][0]], o = meanSur(orig[i]);
    for (let j = 1; j < cands[i].length; j++) {
      check(); onStep?.(`Замеряю варианты: ${++done} из ${total}`);
      const sc = await scoreSentence(doc.sents[i - 1] || "", cands[i][j]);
      // fluency guard: a variant far less likely than the original is broken text, not "human" text
      if (meanSur(sc) <= Math.max(o * 1.8, o + 2)) { scored[i].push(sc); keep.push(cands[i][j]); }
    }
    cands[i] = keep;
  }

  // 5. coordinate descent on the assembled profile
  onStep?.("Подбираю лучшее сочетание");
  const dist = p => target ? distance(p, target) : typicalityDistance(p);
  const assemble = ch => profileOf(ch.map((c, i) => scored[i][c]), lang);
  let choice = new Array(n).fill(0), best = dist(assemble(choice));
  const start = best;
  for (let sweep = 0; sweep < 4; sweep++) {
    let improved = false;
    for (let i = 0; i < n; i++) for (let j = 0; j < scored[i].length; j++) {
      if (j === choice[i]) continue;
      const trial = choice.slice(); trial[i] = j;
      const d = dist(assemble(trial));
      if (d < best - 1e-9) { best = d; choice = trial; improved = true; }
    }
    if (!improved) break;
  }
  const outText = doc.render(choice.map((c, i) => cands[i][c]));
  return {
    text: outText,
    before: profileOf(orig, lang),
    after: assemble(choice),
    distBefore: start, distAfter: best,
    changed: choice.filter(Boolean).length, n,
    candidates: cands.map(c => c.length - 1),
    targetMode: target ? "style" : "typicality",
  };
}
