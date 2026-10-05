import * as core from "./core.js";

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
core.configure({ local: params.has("selftest") });

const EXAMPLE = "Искусственный интеллект играет ключевую роль в современном мире. Он позволяет автоматизировать рутинные задачи и значительно повышает эффективность работы компаний. Кроме того, технологии машинного обучения открывают новые возможности для анализа больших объёмов данных. Важно отметить, что внедрение ИИ требует тщательного планирования и подготовки сотрудников. Компании, которые инвестируют в обучение персонала, получают значительное конкурентное преимущество. Однако, существуют и определённые риски, связанные с использованием искусственного интеллекта. Среди них можно выделить вопросы конфиденциальности данных и этические аспекты принятия решений. Таким образом, искусственный интеллект является мощным инструментом, который при правильном использовании может принести огромную пользу обществу.";

const src = $("src"), out = $("out"), status = $("status");
let result = null, ctl = null, styleTarget = null, styleKey = "", seed = 1;
const store = {
  get: k => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch {} },
};
const val = name => document.querySelector(`input[name="${name}"]:checked`).value;
const langOf = t => (val("lang") === "auto" ? core.detectLang(t) : val("lang"));
const setStatus = (t, err) => { status.textContent = t; status.classList.toggle("err", !!err); };
const counts = () => {
  $("src-count").textContent = core.words(src.value).length + " сл.";
  $("out-count").textContent = result ? core.words(result.text).length + " сл." : "";
};

// ---------------- metrics view
const METRICS = [
  { k: "fdg", name: "Fast-DetectGPT", hint: "Насколько текст вероятнее типичного для модели. У ИИ высоко, у людей около нуля", fmt: v => v.toFixed(2), max: 10, dir: -1, abs: true },
  { k: "top1", name: "Слова = первая догадка модели", hint: "GLTR: доля слов, которые модель угадала с первой попытки", fmt: v => Math.round(v * 100) + "%", max: 1, dir: -1 },
  { k: "top10", name: "Слова из топ-10 догадок", hint: "У текстов ИИ почти все слова отсюда", fmt: v => Math.round(v * 100) + "%", max: 1, dir: -1 },
  { k: "burst", name: "Ритм «удивления»", hint: "Разброс предсказуемости между предложениями (burstiness)", fmt: v => v.toFixed(2), max: 0.8, dir: +1 },
  { k: "lenCv", name: "Разброс длины предложений", hint: "Люди пишут то коротко, то длинно", fmt: v => v.toFixed(2), max: 1, dir: +1 },
  { k: "clPer100", name: "Штампы на 100 слов", hint: "«Важно отметить», «Кроме того», «Moreover»…", fmt: v => v.toFixed(1), max: 8, dir: -1 },
];
function renderMetrics(b, a) {
  const box = $("metrics"); box.innerHTML = "";
  for (const m of METRICS) {
    const bv = b ? b[m.k] : null, av = a ? a[m.k] : null;
    let cls = "";
    if (b && a) { const d = ((m.abs ? Math.abs(av) - Math.abs(bv) : av - bv)) * m.dir; cls = d > 1e-6 ? "delta-good" : d < -1e-6 ? "delta-bad" : ""; }
    const w = v => v === null ? "0%" : Math.min(100, Math.abs(v) / m.max * 100).toFixed(1) + "%";
    const el = document.createElement("div"); el.className = "metric";
    el.innerHTML = `<div class="name"></div><div class="hint"></div><div class="bars">
      <span>до</span><div class="bar"><i style="width:${w(bv)}"></i></div><span class="val">${bv === null ? "–" : m.fmt(bv)}</span>
      <span>после</span><div class="bar after"><i style="width:${w(av)}"></i></div><span class="val ${cls}">${av === null ? "–" : m.fmt(av)}</span></div>`;
    el.querySelector(".name").textContent = m.name; el.querySelector(".hint").textContent = m.hint;
    box.append(el);
  }
}

// ---------------- model loading
const fmtMB = b => (b / 1048576).toFixed(0) + " МБ";
async function loadModels() {
  $("load").disabled = true; $("progress").hidden = false;
  $("model-title").textContent = "Скачиваю нейросеть…";
  try {
    try { await navigator.storage?.persist?.(); } catch {}
    const info = await core.loadModels(({ loaded, total }) => {
      if (total) {
        $("progress").firstElementChild.style.width = (loaded / total * 100).toFixed(1) + "%";
        $("model-sub").textContent = `${fmtMB(loaded)} из ${fmtMB(total)}`;
      }
    });
    $("progress").hidden = true; $("load").hidden = true;
    $("model-title").textContent = "Нейросеть на устройстве";
    $("model-sub").textContent = info.embedder ? "Работает без интернета. Смысл проверяется отдельной моделью." : "Работает без интернета. Модель проверки смысла не загрузилась, варианты не фильтруются.";
    $("model-pill").hidden = false; $("model-pill").className = "pill ok";
    $("model-pill").textContent = info.device === "webgpu" ? "WebGPU" : "Процессор";
    $("run").disabled = false;
    store.set("humanizer-models", "1");
  } catch (e) {
    console.error(e);
    $("load").disabled = false; $("progress").hidden = true;
    $("model-title").textContent = "Не удалось загрузить нейросеть";
    $("model-sub").textContent = navigator.onLine ? "Проверьте свободное место и попробуйте ещё раз." : "Для первой загрузки нужен интернет. Потом приложение работает офлайн.";
  }
}
$("load").onclick = loadModels;

// ---------------- style sample
function renderStyleState() {
  const t = $("style").value.trim(), n = core.words(t).length;
  if (!t) { $("style-state").textContent = "необязательно: вставьте свои прошлые тексты, и результат будет подогнан под ваш почерк"; $("style-summary").textContent = ""; }
  else if (n < 150) { $("style-state").textContent = "образец слишком короткий"; $("style-summary").textContent = `${n} сл. Нужно хотя бы 150, лучше 300+.`; }
  else { $("style-state").textContent = "включён: результат подгоняется под ваш стиль"; $("style-summary").textContent = styleTarget ? summaryOf(styleTarget) : `${n} сл. Профиль построится при запуске.`; }
  $("dist-label").textContent = n >= 150 ? "Отклонение от вашего стиля" : "Отклонение от человеческого профиля";
}
const summaryOf = p => `${p.words} сл. · первая догадка ${Math.round(p.top1 * 100)}% · Fast-DetectGPT/токен ${p.fdgPerTok.toFixed(3)} · ритм ${p.burst.toFixed(2)} · разброс длины ${p.lenCv.toFixed(2)}`;
async function ensureStyle() {
  const t = $("style").value.trim();
  if (core.words(t).length < 150) return null;
  if (styleTarget && styleKey === t) return styleTarget;
  styleTarget = await core.styleProfile(t, s => setStatus(s));
  styleKey = t;
  renderStyleState();
  return styleTarget;
}

// ---------------- run
$("run").onclick = async () => {
  const text = src.value.trim();
  if (!text) { setStatus("Вставьте текст слева.", true); return; }
  ctl = new AbortController();
  $("run").disabled = true; $("stop").hidden = false; $("copy").disabled = true;
  out.className = "out thinking"; out.textContent = "Работаю…";
  try {
    const target = await ensureStyle();
    const lang = langOf(text);
    result = await core.humanize(text, { lang, target, paraphrases: +val("mode"), seed: seed++, signal: ctl.signal, onStep: s => setStatus(s) });
    result.text = core.removeCliches(result.text, lang, core.rng(seed));
    out.className = "out"; out.textContent = result.text;
    $("idx-before").textContent = result.distBefore.toFixed(1); $("idx-after").textContent = result.distAfter.toFixed(1);
    renderMetrics(result.before, result.after);
    setStatus(`Готово: изменено предложений ${result.changed} из ${result.n}.`);
    $("copy").disabled = false;
  } catch (e) {
    if (e?.name === "AbortError") { setStatus("Остановлено."); out.className = "out placeholder"; out.textContent = result ? result.text : "Здесь появится переписанный текст."; }
    else { console.error(e); setStatus("Что-то пошло не так: " + (e?.message || e), true); out.className = "out placeholder"; out.textContent = "Не получилось. Попробуйте режим «Быстро» или текст покороче."; }
  } finally { $("run").disabled = false; $("stop").hidden = true; counts(); }
};
$("stop").onclick = () => ctl?.abort();
$("clear").onclick = () => { src.value = ""; result = null; out.className = "out placeholder"; out.textContent = "Здесь появится переписанный текст."; $("copy").disabled = true; renderMetrics(null, null); counts(); store.set("humanizer-draft", ""); src.focus(); };
$("copy").onclick = async () => {
  try { await navigator.clipboard.writeText(result.text); setStatus("Скопировано."); }
  catch { const r = document.createRange(); r.selectNodeContents(out); const s = getSelection(); s.removeAllRanges(); s.addRange(r); setStatus("Текст выделен: скопируйте его вручную."); }
};
let t1, t2;
src.addEventListener("input", () => { clearTimeout(t1); t1 = setTimeout(() => { counts(); store.set("humanizer-draft", src.value); }, 250); });
$("style").addEventListener("input", () => { clearTimeout(t2); t2 = setTimeout(() => { renderStyleState(); store.set("humanizer-style", $("style").value); }, 250); });

// ---------------- boot
src.value = store.get("humanizer-draft") || EXAMPLE;
$("style").value = store.get("humanizer-style") || "";
if ($("style").value.trim()) $("style-box").open = true;
renderStyleState(); renderMetrics(null, null); counts();
if (store.get("humanizer-models") || params.has("selftest")) loadModels();   // cached: loads from disk, works offline
if ("serviceWorker" in navigator && !params.has("selftest")) navigator.serviceWorker.register("sw.js").catch(() => {});
window.__humanizer = { core, run: () => $("run").click() };
