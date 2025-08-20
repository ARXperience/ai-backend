/* app.js – Studio Chatbot v2 (GENÉRICO)
/*
  ✅ Características:
  - RAG local (TF-IDF + expansión de sinónimos)
  - Ingesta de archivos (.txt, .md, .csv, .json, .jsonl, .html, .rtf, .pdf con pdfjs)
  - Ingesta por URL (limpieza agresiva segura + extracción de datos clave)
  - Wizard JSONL: autocompletar desde URL, edición, previsualización, descarga y “agregar al proyecto”
  - Historial de conversación (tester y mini widget) persistente en localStorage
  - Fallback a servidor de IA (ej. tu backend con Gemini) con historial + contexto
  - Modo Always-On (nunca se queda sin responder)
  - Exportar HTML estático básico (snippet)

  🔧 Cambia esta URL por tu backend (Node/Express u otro) que acepte POST /chat
*/
const AI_SERVER_URL = "https://TU_BACKEND_DE_IA/chat";

/* ===================== Estado global ===================== */
const state = {
  bot: { name:"", goal:"", notes:"", system:"", topk:5, threshold:0.15 },
  sources: /** @type {Array<Source>} */ ([]),
  docs:    /** @type {Array<Doc>} */   ([]),
  index:   { vocab:new Map(), idf:new Map(), built:false },
  urlsQueue: [],
  chat: [],       // historial del tester
  miniChat: [],   // historial del mini widget
  qa: /** @type {Array<{q:string,a:string,src?:string,tags?:string[]}>} */([]),
  settings: { allowWeb: true, strictContext: true },
  metaDocId: null // doc sintético con (name, goal, notes, system)
};

let ingestBusy = false;

/* ===================== Tipos JSDoc ===================== */
/**
 * @typedef {{id:string, type:'file'|'url', title:string, href?:string, addedAt:number}} Source
 * @typedef {{id:string, sourceId:string, title:string, text:string, chunks:Array<Chunk>}} Doc
 * @typedef {{id:string, text:string, vector:Map<string, number>}} Chunk
 */

/* ===================== Helpers DOM ===================== */
const $ = (id) => document.getElementById(id);
const el = (tag, attrs={}, children=[])=>{
  const n = document.createElement(tag);
  Object.entries(attrs).forEach(([k,v])=>{
    if (k==='class') n.className = v;
    else if (k==='text') n.textContent = v;
    else n.setAttribute(k,v);
  });
  children.forEach(c => n.appendChild(c));
  return n;
};
const nowId = ()=> Math.random().toString(36).slice(2)+Date.now().toString(36);
const uniq = (arr)=> Array.from(new Set(arr.filter(Boolean)));

/* ===================== Persistencia ===================== */
const STORAGE_KEY = "studio-chatbot-v2";
function save() {
  const toSave = {
    bot: state.bot,
    sources: state.sources,
    docs: state.docs.map(d=>({ id:d.id, sourceId:d.sourceId, title:d.title, text:d.text })), // no guardo vectors/chunks
    urlsQueue: state.urlsQueue,
    qa: state.qa,
    chat: state.chat,
    miniChat: state.miniChat,
    settings: state.settings,
    metaDocId: state.metaDocId
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
}
function load() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const data = JSON.parse(raw);
    Object.assign(state.bot, data.bot||{});
    state.sources = data.sources||[];
    state.docs = (data.docs||[]).map(d=>({...d, chunks:[]}));
    state.urlsQueue = data.urlsQueue||[];
    state.qa = data.qa||[];
    state.chat = data.chat || [];
    state.miniChat = data.miniChat || [];
    state.settings = data.settings || state.settings;
    state.metaDocId = data.metaDocId || null;
  } catch(e){ console.warn("No se pudo cargar estado:", e); }
}

/* ===================== Utilidades de texto ===================== */
const STOP = new Set(("a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos el ella ellas ellos en entre era erais éramos eran es esa esas ese esos esta estaba estabais estábamos estaban estar este esto estos fue fui fuimos ha han hasta hay la las le les lo los mas más me mientras muy nada ni nos o os otra otros para pero poco por porque que quien se ser si sí sin sobre soy su sus te tiene tengo tuvo tuve u un una unas unos y ya").split(/\s+/));

function normalizeText(t){
  return (t||"")
    .replace(/<script[\s\S]*?<\/script>/gi," ")
    .replace(/<style[\s\S]*?<\/style>/gi," ")
    .replace(/<[^>]+>/g," ")
    .replace(/&[a-z]+;/gi," ")
    .replace(/\s+/g," ")
    .trim();
}

function tokens(text){
  return (text||"")
    .toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g,"")
    .replace(/[^a-z0-9áéíóúñü\s]/gi, ' ')
    .split(/\s+/)
    .filter(w => w && !STOP.has(w) && w.length>1);
}

function chunkText(text, chunkSize=1200, overlap=120){
  const words = (text||"").split(/\s+/);
  const chunks = [];
  const step = Math.max(1, Math.floor((chunkSize - overlap)));
  for (let i=0;i<words.length;i+=step){
    const part = words.slice(i, i+chunkSize).join(' ').trim();
    if (part.length<40) continue;
    chunks.push(part);
  }
  return chunks;
}

function topSentences(text, max=3, minLen=40){
  const sents = (text||"")
    .replace(/\s+/g," ")
    .split(/(?<=[\.\!\?])\s+/)
    .map(s=>s.trim())
    .filter(s=> s && s.length>=minLen && /[a-zA-Záéíóúñ]/i.test(s));
  // orden aproximado: más largos primero (proxy de “información densa”)
  sents.sort((a,b)=> b.length - a.length);
  return sents.slice(0, max);
}

/* ===================== Indexado TF-IDF ===================== */
function buildIndex(){
  createOrUpdateMetaDoc(); // aseguramos meta-doc actualizado

  const vocab = new Map();
  const allChunks = [];
  state.docs.forEach(doc=>{
    if (!doc.chunks?.length) {
      const cks = chunkText(doc.text);
      doc.chunks = cks.map((t,i)=>({id:`${doc.id}#${i}`, text:t, vector:new Map()}));
    }
    doc.chunks.forEach(ch=>{
      allChunks.push(ch);
      const seen = new Set();
      tokens(ch.text).forEach(tok=>{
        if (!seen.has(tok)){
          vocab.set(tok, (vocab.get(tok)||0)+1);
          seen.add(tok);
        }
      });
    });
  });

  const N = allChunks.length || 1;
  const idf = new Map();
  for (const [term, df] of vocab){
    idf.set(term, Math.log((N+1)/(df+1))+1);
  }

  allChunks.forEach(ch=>{
    const tf = new Map();
    const toks = tokens(ch.text);
    toks.forEach(t=> tf.set(t, (tf.get(t)||0)+1));
    const vec = new Map();
    for (const [t,f] of tf){
      const idf_t = idf.get(t) || 0;
      vec.set(t, (f/toks.length) * idf_t);
    }
    ch.vector = vec;
  });

  state.index.vocab = vocab;
  state.index.idf = idf;
  state.index.built = true;
  renderCorpus();
}

/* ===================== Meta-doc (inyecta objetivo/notas) ===================== */
function createOrUpdateMetaDoc(){
  const { name, goal, notes, system } = state.bot;
  const blocks = [];
  if (name)   blocks.push(`NOMBRE DEL BOT: ${name}`);
  if (goal)   blocks.push(`OBJETIVO: ${goal}`);
  if (notes)  blocks.push(`NOTAS: ${notes}`);
  if (system) blocks.push(`INSTRUCCIONES: ${system}`);
  const text = blocks.join("\n\n");

  if (!text) return;

  if (state.metaDocId){
    // actualiza
    const d = state.docs.find(x=> x.id===state.metaDocId);
    if (d) { d.text = text; d.chunks = []; return; }
  }
  // crea
  const sid = nowId();
  state.sources.push({id:sid, type:'file', title:'(perfil del bot)', addedAt:Date.now()});
  const did = nowId();
  state.docs.push({id:did, sourceId:sid, title:'Perfil del bot', text, chunks:[]});
  state.metaDocId = did;
}

/* ===================== Vector math ===================== */
function cosineSim(a,b){
  let dot=0, na=0, nb=0;
  a.forEach((va, t)=>{ const vb=b.get(t)||0; dot += va*vb; na += va*va; });
  b.forEach(vb=>{ nb += vb*vb; });
  if (na===0 || nb===0) return 0;
  return dot / (Math.sqrt(na)*Math.sqrt(nb));
}

function vectorizeQuery(q){
  const tf = new Map();
  const toks = tokens(q);
  toks.forEach(t=> tf.set(t, (tf.get(t)||0)+1));
  const vec = new Map();
  toks.forEach(t=>{
    const idf_t = state.index.idf.get(t) || 0;
    vec.set(t, (tf.get(t)/toks.length) * idf_t);
  });
  return vec;
}

/* ===================== Búsqueda (TF-IDF + sinónimos) ===================== */
const SYN = {
  precio:["precios","tarifa","costo","valor","cuánto","vale","cotización","cotizar"],
  horario:["horarios","apertura","atención","agenda","disponible","disponibilidad"],
  comprar:["compra","adquirir","pagar","checkout","pedido","carrito"],
  contacto:["whatsapp","teléfono","email","correo","soporte","ayuda"],
  envío:["envíos","delivery","entrega","tiempos","plazo","domicilio"],
  devolución:["devoluciones","cambios","garantía","reembolso","política"],
  servicio:["servicios","productos","oferta","portafolio","planes","paquetes"]
};
function expandQuery(q){
  const base = tokens(q).join(' ');
  let extra = [];
  for (const [k, arr] of Object.entries(SYN)){
    if (new RegExp(`\\b${k}\\b`,"i").test(q)) extra = extra.concat(arr);
  }
  return extra.length ? `${base} ${uniq(extra).join(' ')}` : base;
}

function searchChunks(query, k=3, thr=0.30){
  if (!state.index.built) buildIndex();
  const run = (qq, threshold)=>{
    const qv = vectorizeQuery(qq);
    const scored = [];
    state.docs.forEach(doc=>{
      doc.chunks.forEach(ch=>{
        const s = cosineSim(qv, ch.vector);
        if (s>=threshold) scored.push({chunk:ch, score:s, doc, source: state.sources.find(s=>s.id===doc.sourceId)});
      });
    });
    scored.sort((a,b)=> b.score - a.score);
    return scored.slice(0, k);
  };
  let hits = run(query, thr);
  if (!hits.length){
    const qx = expandQuery(query);
    hits = run(qx, Math.max(0.10, thr*0.7)); // baja umbral con sinónimos
  }
  return hits;
}

/* ===================== Q&A JSONL (match semántico previo) ===================== */
function simQ(a,b){ return cosineSim(vectorizeQuery(a), vectorizeQuery(b)); }

function answerFromQA(query){
  if (!state.qa.length) return null;
  let best = {i:-1, score:0};
  for (let i=0;i<state.qa.length;i++){
    const s = simQ(query, state.qa[i].q);
    if (s > best.score) best = {i, score:s};
  }
  // Umbral semántico Q&A (laxo 0.30)
  return (best.score >= 0.30) ? state.qa[best.i] : null;
}

/* ===================== Always-On: intención + respuesta generada ===================== */
function guessIntent(q){
  const s = (q||"").toLowerCase();
  if (/(precio|costo|vale|cu[aá]nto).*(plan|servicio)|\bplanes?\b/.test(s)) return 'planes';
  if (/(afili|inscrib|registro|suscrip|alta)/.test(s)) return 'afiliacion';
  if (/(whats?app|contact|tel[eé]fono|celular|direcci[oó]n|ubicaci[oó]n|correo|email)/.test(s)) return 'contacto';
  if (/(pol[ií]tica|garant[ií]a|devoluci[oó]n|reembolso|t[eé]rminos|condiciones)/.test(s)) return 'politicas';
  if (/(horario|atenci[oó]n|hora|agenda|cita)/.test(s)) return 'agenda';
  return 'general';
}
function generateAlwaysOnAnswer(q){
  switch (guessIntent(q)) {
    case 'planes':
      return "Puedo detallar planes, precios y qué incluye cada opción. Dime qué necesitas y tu presupuesto aproximado para recomendarte mejor.";
    case 'afiliacion':
      return "Te ayudo a suscribirte: necesito tu nombre, correo y el plan/servicio que prefieres. Te guío paso a paso.";
    case 'contacto':
      return "¿Prefieres WhatsApp, teléfono o email? Te comparto los datos de contacto y, si quieres, dejo registro de tu solicitud.";
    case 'politicas':
      return "Te explico políticas de garantía, cambios y devoluciones con los plazos y requisitos. ¿Qué caso quieres resolver?";
    case 'agenda':
      return "¿Qué día y franja te sirve? Propongo horarios y confirmo la cita.";
    default:
      return "Puedo ayudarte con información de servicios, precios, horarios, políticas y contacto. Cuéntame tu objetivo para darte una respuesta concreta.";
  }
}

/* ===================== Lectura de archivos / URLs ===================== */
async function readFileAsText(file){
  const ext = (file.name.split('.').pop()||"").toLowerCase();
  if (['txt','md','csv','json','html','htm','rtf','jsonl'].includes(ext)){
    const raw = await file.text();
    if (ext==='json') return JSON.stringify(JSON.parse(raw), null, 2);
    if (ext==='html' || ext==='htm') return normalizeText(raw);
    return raw;
  }
  if (ext==='pdf'){
    if (window.pdfjsLib){
      const buf = await file.arrayBuffer();
      const pdf = await window.pdfjsLib.getDocument({data:buf}).promise;
      let out="";
      for (let i=1;i<=pdf.numPages;i++){
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const pageText = content.items.map(it=> it.str).join(" ");
        out += pageText + "\n";
      }
      return out;
    } else {
      alert("Para leer PDF local, incluye pdfjs (pdfjsLib) o sube el texto en .txt/.md.");
      return "";
    }
  }
  alert(`Formato no soportado nativamente: .${ext}. Convierte a .txt/.md/.pdf (con pdfjs).`);
  return "";
}

async function fetchUrlText(url){
  try{
    const res = await fetch(url, {mode:'cors'});
    const ct = res.headers.get('content-type')||"";
    const raw = await res.text();
    if (ct.includes("html")) return raw; // devolvemos RAW (limpieza se hace aparte)
    return raw;
  }catch(e){
    console.warn("CORS o fetch falló:", e);
    return "";
  }
}

/* ===================== LIMPIEZA y EXTRACCIÓN (versiones mejoradas) ===================== */

// LIMPIEZA AGRESIVA pero segura (conserva "contacto" real)
function cleanForAnswer(input){
  let t = (input||"");

  // Quita URLs (no queremos que contaminen respuestas)
  t = t.replace(/https?:\/\/\S+/gi, " ");

  // Encabezados típicos de scrapers
  t = t.replace(/^(Title|URL Source|Published Time|Markdown Content|Image \d+):.*$/gmi, " ");

  // Markdown: imágenes y links
  t = t.replace(/!\[[^\]]*\]\([^)]+\)/g, " ");
  t = t.replace(/\[[^\]]*\]\([^)]+\)/g, " ");

  // Quita HTML y entidades
  t = t.replace(/<script[\s\S]*?<\/script>/gi, " ")
       .replace(/<style[\s\S]*?<\/style>/gi, " ")
       .replace(/<[^>]+>/g, " ")
       .replace(/&[a-z]+;/gi, " ");

  // Quita avisos de cookies / UI muy genéricos (línea completa)
  t = t.replace(/este sitio utiliza cookies.*?acept(a|o)?|accept|decline/gi, " ");

  // Elimina LÍNEAS DE MENÚ (pero NO borra la palabra "contacto" dentro de frases completas)
  const MENU_WORDS = ["inicio","home","blog","tienda","shop","carrito","cart","mi cuenta","account","nosotros","about","servicios","productos","contacto","contact"];
  t = t.split(/\n+/).map(line=>{
    const l = line.trim().toLowerCase();
    // detecta listas muy cortas con sólo palabras de menú y separadores
    const isMenuLike =
      l.length <= 80 &&
      !/[\.!\?@0-9]/.test(l) &&
      /(\||•|·|\/|>|-|\s{2,})/.test(l) &&
      l.split(/[\s\|•·>\/-]+/).filter(Boolean).every(w => MENU_WORDS.includes(w));
    return isMenuLike ? "" : line;
  }).filter(Boolean).join("\n");

  // Espacios
  t = t.replace(/[ \t]+/g, " ").replace(/\s{2,}/g, " ").trim();
  return t;
}

// Extrae datos estructurados desde RAW (para contactos) y CLEAN (para descripciones)
function extractFromText(url, rawText){
  const raw = rawText || "";
  const text = cleanForAnswer(raw);

  const linesRaw  = raw.split(/\n+/).map(s=>s.trim()).filter(Boolean);
  const linesClean= text.split(/\n+/).map(s=>s.trim()).filter(Boolean);

  // Emails (desde RAW)
  const emails = Array.from((raw||"").matchAll(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi)).map(m=>m[0]);

  // Teléfonos (desde RAW, normalizados; también detecta wa.me)
  const wa = Array.from((raw||"").matchAll(/wa\.me\/(\d{7,15})/gi)).map(m=>m[1]);
  const phoneCandidates = Array.from((raw||"").matchAll(/\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b/g))
    .map(m=>m[0]);
  const phones = Array.from(new Set(
    wa.concat(
      phoneCandidates.map(p=>p.replace(/[^\d+]/g,""))
    ).filter(d=>{
      const digits = d.replace(/\D/g,"");
      const len = digits.length;
      return len>=7 && len<=15;
    })
  ));

  // Horarios / ofertas / políticas (desde CLEAN para reducir ruido)
  const hours = linesClean.filter(l=>/(horario|lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|\b\d{1,2}:\d{2}\b|\bam\b|\bpm\b)/i.test(l)).slice(0,8);
  const offers = linesClean.filter(l=>/(precio|plan|paquete|servicio|producto|\$|\bUSD\b|\bCOP\b|\bMXN\b)/i.test(l)).slice(0,12);
  const policies = linesClean.filter(l=>/(pol[ií]tica|t[eé]rminos|condiciones|garant[ií]a|devoluci[oó]n|reembolso|privacidad)/i.test(l)).slice(0,12);

  // Descripción compacta (frases limpias y largas)
  const desc = topSentences(text, 4, 50).join(" ");

  // FAQs (patrón Q? + siguiente como respuesta; desde CLEAN)
  const faqs = [];
  for (let i=0;i<linesClean.length-1;i++){
    const q = linesClean[i], a = linesClean[i+1];
    if (/\?/.test(q) && a && !/\?$/.test(a)) {
      const ans = topSentences(a, 2, 30).join(" ");
      if (ans) faqs.push({ q: q.replace(/\s+/g," ").trim(), a: ans });
    }
  }

  // Contacto legible (emails/teléfonos + pistas desde RAW y CLEAN)
  const contactHints = [];
  const contactRaw = linesRaw.filter(l=>/(whats?app|contacto|direcci[oó]n|ubicaci[oó]n|soporte|correo|email|tel[eé]fono|celular)/i.test(l)).slice(0,3);
  const contactClean = linesClean.filter(l=>/(whats?app|contacto|direcci[oó]n|ubicaci[oó]n|soporte|correo|email|tel[eé]fono|celular)/i.test(l)).slice(0,3);
  if (emails.length) contactHints.push(`Email(s): ${uniq(emails).join(", ")}`);
  if (phones.length) contactHints.push(`Teléfono(s): ${uniq(phones).join(", ")}`);
  contactHints.push(...contactRaw, ...contactClean);
  const contact = uniq(contactHints).filter(Boolean).slice(0,4).join(" • ");

  // Nombre aproximado por dominio
  let name=""; try { const host = new URL(url).hostname.replace(/^www\./,''); name = host.split('.')[0]; } catch {}
  name = name ? (name.charAt(0).toUpperCase()+name.slice(1)) : "";

  return { name, desc, contact, hours, offers, policies, faqs };
}

/* ===================== Ingesta ===================== */
async function ingestFiles(files){
  if (!files?.length) return;
  setBusy(true);
  const bar = $("ingestProgress");
  if (bar) bar.style.width = "0%";
  let done = 0;

  for (const f of files){
    const ext = (f.name.split('.').pop()||"").toLowerCase();

    // Soporte *.jsonl (Q&A programadas)
    if (ext === 'jsonl'){
      const raw = await f.text();
      const lines = raw.split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
      const qaPairs = [];
      for (const line of lines){
        try{
          const obj = JSON.parse(line);
          if (obj && obj.q && obj.a){
            qaPairs.push({ q:String(obj.q), a:String(obj.a), src: obj.src?String(obj.src):undefined, tags: Array.isArray(obj.tags)?obj.tags:undefined });
          }
        }catch{}
      }
      if (qaPairs.length){
        state.qa.push(...qaPairs);
        // También indexamos como texto para RAG
        const txt = qaPairs.map(x=>`PREGUNTA: ${x.q}\nRESPUESTA: ${x.a}${x.src?`\nFUENTE: ${x.src}`:""}`).join("\n\n");
        const sid = nowId();
        state.sources.push({id:sid, type:'file', title:f.name, addedAt:Date.now()});
        state.docs.push({id:nowId(), sourceId:sid, title:f.name, text:txt, chunks:[]});
      }
      done++;
      if (bar) bar.style.width = `${Math.round(done/files.length*100)}%`;
      continue;
    }

    // Otros formatos
    const text = await readFileAsText(f);
    if (!text) { done++; if (bar) bar.style.width = `${Math.round(done/files.length*100)}%`; continue; }

    const sourceId = nowId();
    state.sources.push({id:sourceId, type:'file', title:f.name, addedAt:Date.now()});
    state.docs.push({id:nowId(), sourceId:sourceId, title:f.name, text, chunks:[]});
    done++;
    if (bar) bar.style.width = `${Math.round(done/files.length*100)}%`;
  }

  buildIndex();
  save();
  renderSources();
  setBusy(false);
  $("modelStatus").textContent = "Con conocimiento";
}

async function ingestUrls(urls){
  if (!urls?.length) return;
  setBusy(true);
  for (let i=0;i<urls.length;i++){
    const u = urls[i];
    const raw = await fetchUrlText(u.url);
    // Guardamos RAW (indexado) + limpiado (mejora de respuesta)
    const cleaned = cleanForAnswer(raw);
    const final = cleaned || normalizeText(raw) || raw || "";
    