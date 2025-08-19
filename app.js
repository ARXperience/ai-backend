/* app.js – Studio Chatbot v2 (genérico, sin marcas)
   ► Índice híbrido (TF-IDF + Jaccard) + re-ranqueo BM25
   ► Expansión de consulta (sinónimos + tolerancia a typos)
   ► Análisis semántico → KB (contacto, horarios, precios, políticas, etc.)
   ► Respuestas claras (saludo + bullets + CTA)
   ► Wizard JSONL: autocompletar por URL, pares canónicos y limpios, export/ingesta
   ► Export HTML estático “offline”
*/

/* ===== Backend IA opcional (déjalo vacío si no usas) ===== */
const AI_SERVER_URL = ""; // ej: "https://api.tu-dominio.com/chat"

/* ===================== Estado ===================== */
const state = {
  bot: { name:"", goal:"", notes:"", system:"", topk:5, threshold:0.15 },
  sources: [],
  docs: [],
  index: {
    vocab:new Map(),   // df por término (stem)
    idf:new Map(),     // idf TF-IDF
    idfBM25:new Map(), // idf BM25
    built:false,
    N:0,
    avgdl:0
  },
  urlsQueue: [],
  chat: [],
  miniChat: [],
  qa: [],
  settings: { allowWeb: true, strictContext: true },
  kb: {
    contact: [], hours: [], prices: [], policies: [],
    products: [], services: [], locations: [],
    faqs: [], sentences: [], keyphrases: new Map()
  },
  vocabSet: new Set()
};

/* ===================== Utils DOM ===================== */
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

/* ===================== Persistencia ===================== */
const STORAGE_KEY = "studio-chatbot-v2";
function save(){
  const toSave = {
    bot: state.bot,
    sources: state.sources,
    docs: state.docs.map(d=>({ id:d.id, sourceId:d.sourceId, title:d.title, text:d.text, meta: !!d.meta })),
    urlsQueue: state.urlsQueue,
    qa: state.qa,
    chat: state.chat,
    miniChat: state.miniChat,
    settings: state.settings,
    kb: state.kb
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
}
function load(){
  const raw = localStorage.getItem(STORAGE_KEY); if (!raw) return;
  try{
    const data = JSON.parse(raw);
    Object.assign(state.bot, data.bot||{});
    state.sources = data.sources||[];
    state.docs = (data.docs||[]).map(d=>({...d, chunks:[], meta: !!d.meta}));
    state.urlsQueue = data.urlsQueue||[];
    state.qa = data.qa||[];
    state.chat = data.chat||[];
    state.miniChat = data.miniChat||[];
    state.settings = Object.assign({ allowWeb:true, strictContext:true }, data.settings||{});
    if (data.kb) state.kb = data.kb;
  }catch(e){ console.warn("No se pudo cargar estado:", e); }
}

/* ===================== NLP básico ===================== */
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
function stripAcc(s){ return s.normalize('NFD').replace(/[\u0300-\u036f]/g,""); }
function tokens(text){
  return stripAcc(text.toLowerCase())
    .replace(/[^a-z0-9áéíóúñü\s]/gi, ' ')
    .split(/\s+/)
    .filter(w => w && !STOP.has(w) && w.length>1);
}
function stemEs(w){
  let s = w;
  s = s.replace(/(mente|ciones|cion|idades|idad|osos?|osas?|ando|iendo|ados?|idas?)$/,'');
  s = s.replace(/(es|s)$/,'');
  return s;
}
function chunkText(text, chunkSize=1200, overlap=120){
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i=0;i<words.length;i+=Math.max(1, Math.floor((chunkSize-overlap)))){
    const part = words.slice(i, i+chunkSize).join(' ').trim();
    if (part.length<40) continue;
    chunks.push(part);
  }
  return chunks;
}

/* ===================== Índice TF-IDF + BM25 ===================== */
function upsertMetaDoc(){
  // Re-crear doc con perfil del bot (goal/notes/system)
  state.docs = state.docs.filter(d=> !d.meta);
  state.sources = state.sources.filter(s=> s.type!=='meta');
  const pieces = [];
  if (state.bot.goal) pieces.push(`OBJETIVO:\n${state.bot.goal}`);
  if (state.bot.notes) pieces.push(`NOTAS:\n${state.bot.notes}`);
  if (state.bot.system) pieces.push(`SISTEMA:\n${state.bot.system}`);
  if (!pieces.length) return;
  const sid = nowId();
  state.sources.push({ id:sid, type:'meta', title:'Perfil del bot', addedAt:Date.now() });
  state.docs.push({ id:nowId(), sourceId:sid, title:'Perfil del bot', text:pieces.join("\n\n"), meta:true, chunks:[] });
}
function buildIndex(){
  upsertMetaDoc();

  const vocabDF = new Map(); // df por término (stem)
  const allChunks = [];
  let totalLen = 0;

  // Construir chunks + tf + vec
  state.docs.forEach(doc=>{
    if (!doc.chunks?.length) {
      const cks = chunkText(doc.text);
      doc.chunks = cks.map((t,i)=>({id:`${doc.id}#${i}`, text:t, vector:new Map(), tf:new Map(), len:0}));
    }
    doc.chunks.forEach(ch=>{
      const toks = tokens(ch.text).map(stemEs);
      ch.len = toks.length;
      totalLen += ch.len;
      // tf
      const tf = new Map();
      toks.forEach(t=> tf.set(t,(tf.get(t)||0)+1));
      ch.tf = tf;
      // df (1 por término en el chunk)
      const seen = new Set();
      toks.forEach(t=>{ if(!seen.has(t)){ vocabDF.set(t,(vocabDF.get(t)||0)+1); seen.add(t);} });
      allChunks.push(ch);
    });
  });

  const N = allChunks.length || 1;
  const avgdl = totalLen / N;

  // IDF TF-IDF y IDF BM25
  const idf = new Map();
  const idfBM25 = new Map();
  for (const [term, df] of vocabDF){
    idf.set(term, Math.log((N+1)/(df+1))+1);
    idfBM25.set(term, Math.log( ( (N - df + 0.5) / (df + 0.5) ) + 1 ));
  }

  // Vector TF-IDF por chunk
  allChunks.forEach(ch=>{
    const vec = new Map();
    for (const [t,f] of ch.tf){
      const idf_t = idf.get(t) || 0;
      vec.set(t, (f / Math.max(1,ch.len)) * idf_t);
    }
    ch.vector = vec;
  });

  state.index.vocab = vocabDF;
  state.index.idf = idf;
  state.index.idfBM25 = idfBM25;
  state.index.built = true;
  state.index.N = N;
  state.index.avgdl = avgdl;

  state.vocabSet = new Set(Array.from(vocabDF.keys()));

  renderCorpus();
  runSemanticAnalysis();
  save();
}

/* ===================== Expansión, similitudes, BM25 ===================== */
const SYN = {
  precio:["costo","tarifa","valor","vale","cuanto","cotizacion","presupuesto","precios","costos","tarifas"],
  horario:["hora","apertura","atencion","cierre","dias","sabado","domingo"],
  contacto:["whatsapp","telefono","llamar","celular","correo","email","direccion","ubicacion","soporte"],
  envio:["entrega","shipping","despacho","reparto","mensajeria","domicilio","tracking","seguimiento"],
  garantia:["garantia","cambios","devolucion","reembolso","tyc","terminos","condiciones","politica","privacidad"],
  producto:["servicio","oferta","plan","paquete","catalogo","portafolio"],
  ubicacion:["sede","oficina","tienda","local","ciudad","pais","zona","cobertura"],
  pago:["pagar","medio","metodo","credito","debito","transferencia","efecty","paypal"],
  soporte:["ayuda","asistencia","ticket","incidencia","reclamo","pqrs"]
};
function getSynonyms(t){
  const out = new Set([t]);
  Object.keys(SYN).forEach(k=>{
    if (k===t || SYN[k].includes(t)) SYN[k].forEach(x=>out.add(x));
  });
  if (t.endsWith('s')) out.add(t.replace(/s$/,'')); else out.add(t+'s');
  out.add(stripAcc(t));
  return Array.from(out).map(stemEs);
}
function levenshtein(a,b){
  a = stripAcc(a); b = stripAcc(b);
  const m = Array(a.length+1).fill(0).map(()=>Array(b.length+1).fill(0));
  for (let i=0;i<=a.length;i++) m[i][0]=i;
  for (let j=0;j<=b.length;j++) m[0][j]=j;
  for (let i=1;i<=a.length;i++){
    for (let j=1;j<=b.length;j++){
      const cost = a[i-1]===b[j-1]?0:1;
      m[i][j] = Math.min(m[i-1][j]+1, m[i][j-1]+1, m[i-1][j-1]+cost);
    }
  }
  return m[a.length][b.length];
}
function expandQueryTerms(q){
  const toks = tokens(q).map(stemEs);
  const expanded = new Set();
  toks.forEach(t=>{
    getSynonyms(t).forEach(x=>expanded.add(x));
    // acercar a vocab (difuso)
    let best=null, bestD=3;
    for (const v of state.vocabSet){
      const d = levenshtein(t, v);
      if (d<bestD){ bestD=d; best=v; if(d===0) break; }
    }
    if (best && bestD<=2) expanded.add(best);
  });
  return Array.from(expanded);
}
function vectorizeQuery(q){
  const ex = expandQueryTerms(q);
  const tf = new Map();
  ex.forEach(t=> tf.set(t, (tf.get(t)||0)+1));
  const v = new Map();
  const n = Math.max(1, ex.length);
  ex.forEach(t=>{
    const idf_t = state.index.idf.get(t) || 0.5;
    v.set(t, (tf.get(t)/n) * idf_t);
  });
  return v;
}
function cosineSim(a,b){
  let dot=0, na=0, nb=0;
  a.forEach((va,t)=>{ const vb=b.get(t)||0; dot += va*vb; na += va*va; });
  b.forEach(vb=>{ nb += vb*vb; });
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na)*Math.sqrt(nb));
}
function tokenSet(s){ return new Set(tokens(s).map(stemEs)); }
function jaccardSet(aSet, bSet){
  let inter=0; bSet.forEach(x=>{ if (aSet.has(x)) inter++; });
  const union = aSet.size + bSet.size - inter;
  return union? inter/union : 0;
}

/* ===================== BM25 y búsqueda híbrida ===================== */
const BM25 = { k1:1.2, b:0.75, wHybrid:0.55, wBM25:0.45 };
function bm25Score(queryTerms, ch){
  const avgdl = state.index.avgdl || 1;
  let score = 0;
  for (const t of queryTerms){
    const idf = state.index.idfBM25.get(t) || 0;
    const tf = ch.tf.get(t) || 0;
    const denom = tf + BM25.k1 * (1 - BM25.b + BM25.b * (ch.len / avgdl));
    if (denom>0) score += idf * ( (tf * (BM25.k1 + 1)) / denom );
  }
  return score;
}
function hybridScore(q, ch){
  const qv = vectorizeQuery(q);
  const cos = cosineSim(qv, ch.vector); // [0..1]
  const qset = new Set(expandQueryTerms(q));
  const cset = tokenSet(ch.text);
  const jac = jaccardSet(qset, cset);   // [0..1]
  return (cos*0.7) + (jac*0.3);
}
function searchChunks(query, k=5, thr=0.15, degrade=false){
  if (!state.index.built) buildIndex();

  const qTerms = expandQueryTerms(query);
  const scores = []; // {doc, chunk, sHybrid, sBM25, sFinal}

  // 1) calcular híbrido y BM25 crudo
  state.docs.forEach(doc=>{
    doc.chunks.forEach(ch=>{
      const sH = hybridScore(query, ch);
      const sB = bm25Score(qTerms, ch); // crudo
      scores.push({ doc, chunk:ch, sHybrid:sH, sBM25:sB, sFinal:0 });
    });
  });

  // 2) normalizar BM25 a [0..1] para combinar
  let minB = Infinity, maxB = -Infinity;
  for (const s of scores){ if (s.sBM25 < minB) minB = s.sBM25; if (s.sBM25 > maxB) maxB = s.sBM25; }
  const rangeB = (maxB - minB) || 1;
  for (const s of scores){
    const bNorm = (s.sBM25 - minB) / rangeB;
    s.sFinal = (BM25.wHybrid * s.sHybrid) + (BM25.wBM25 * bNorm);
  }

  // 3) filtrar por umbral y ordenar
  let filtered = scores.filter(x => x.sFinal >= thr);
  filtered.sort((a,b)=> b.sFinal - a.sFinal);

  if (filtered.length) return filtered.slice(0, k);
  if (!degrade){
    // degradación: bajamos umbral y subimos k
    return searchChunks(query, Math.max(k,8), Math.max(0.08, thr*0.66), true);
  }
  return [];
}

/* ===================== Lectores / Fetch ===================== */
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
    } else { alert("Para leer PDF, incluye pdfjs (pdfjsLib) o convierte a .txt/.md."); return ""; }
  }
  alert(`Formato no soportado: .${ext}. Convierte a .txt/.md/.pdf.`); return "";
}
// Rastreo con alternativas por CORS
async function fetchUrlText(url){
  try{
    const r = await fetch(url, { mode:'cors' });
    const ct = (r.headers.get('content-type')||"").toLowerCase();
    const raw = await r.text();
    if (ct.includes("html")) return normalizeText(raw);
    return raw;
  }catch{}
  try{ // Jina reader
    const cleanURL = url.replace(/^https?:\/\//,'');
    const r = await fetch(`https://r.jina.ai/http://${cleanURL}`);
    const raw = await r.text(); if (raw && raw.length>50) return normalizeText(raw);
  }catch{}
  try{ // AllOrigins
    const r = await fetch(`https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`);
    const raw = await r.text(); return normalizeText(raw);
  }catch(e){ console.warn("No se pudo rastrear URL:", url, e); return ""; }
}

/* ===================== Ingesta ===================== */
async function ingestFiles(files){
  if (!files?.length) return;
  setBusy(true);
  const bar = $("ingestProgress"); if (bar) bar.style.width="0%";
  let done = 0;

  for (const f of files){
    const ext = (f.name.split('.').pop()||"").toLowerCase();

    if (ext === 'jsonl'){
      const raw = await f.text();
      const lines = raw.split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
      const qaPairs = [];
      for (const line of lines){
        try{
          const obj = JSON.parse(line);
          if (obj && obj.q && obj.a) qaPairs.push({ q:String(obj.q), a:String(obj.a), src: obj.src?String(obj.src):undefined });
        }catch{}
      }
      if (qaPairs.length){
        state.qa.push(...qaPairs);
        const txt = qaPairs.map(x=>`PREGUNTA: ${x.q}\nRESPUESTA: ${x.a}${x.src?`\nFUENTE: ${x.src}`:""}`).join("\n\n");
        const sid = nowId();
        state.sources.push({id:sid, type:'file', title:f.name, addedAt:Date.now()});
        state.docs.push({id:nowId(), sourceId:sid, title:f.name, text:txt, chunks:[]});
      }
      done++; if (bar) bar.style.width = `${Math.round(done/files.length*100)}%`; continue;
    }

    const text = await readFileAsText(f);
    if (!text) { done++; if (bar) bar.style.width = `${Math.round(done/files.length*100)}%`; continue; }

    const sourceId = nowId();
    state.sources.push({id:sourceId, type:'file', title:f.name, addedAt:Date.now()});
    state.docs.push({id:nowId(), sourceId:sourceId, title:f.name, text, chunks:[]});
    done++; if (bar) bar.style.width = `${Math.round(done/files.length*100)}%`;
  }

  buildIndex(); save(); renderSources(); setBusy(false);
  const ms = $("modelStatus"); if (ms) ms.textContent = "Con conocimiento";
}
async function ingestUrls(urls){
  if (!urls?.length) return;
  setBusy(true);
  for (let i=0;i<urls.length;i++){
    const u = urls[i];
    const text = await fetchUrlText(u.url);
    const sid = nowId();
    state.sources.push({id:sid, type:'url', title:u.title||u.url, href:u.url, addedAt:Date.now()});
    state.docs.push({id:nowId(), sourceId:sid, title:u.title||u.url, text: text||"", chunks:[]});
  }
  buildIndex(); save(); renderSources(); setBusy(false);
  const ms = $("modelStatus"); if (ms) ms.textContent = "Con conocimiento";
}

/* ===================== Análisis semántico → KB ===================== */
function pushFact(arr, v){ if (v && typeof v==='string'){ const s=v.trim(); if (s && !arr.includes(s)) arr.push(s); } }
function runSemanticAnalysis(){
  state.kb = { contact:[], hours:[], prices:[], policies:[], products:[], services:[], locations:[], faqs:[], sentences:[], keyphrases:new Map() };
  const addKey = (term, w=1)=>{ const k=stemEs(stripAcc(term.toLowerCase())); state.kb.keyphrases.set(k,(state.kb.keyphrases.get(k)||0)+w); };

  state.docs.forEach(doc=>{
    const text = doc.text || "";
    const lines = text.split(/\n+/).map(s=>s.trim()).filter(Boolean);

    // Contacto
    const emails = Array.from((text||"").matchAll(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi)).map(m=>m[0]);
    const phones = Array.from((text||"").matchAll(/(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,3}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}/g)).map(m=>m[0]).filter(x=>x.replace(/\D/g,'').length>=7);
    const contactExtras = lines.filter(l=>/(whats?app|contacto|direcci[oó]n|ubicaci[oó]n|atenci[oó]n|soporte|correo|email|tel[eé]fono|celular)/i.test(l)).slice(0,10);
    const contactStr = [
      emails.length? `Emails: ${[...new Set(emails)].join(", ")}`:"", 
      phones.length? `Teléfonos: ${[...new Set(phones)].join(", ")}`:"", 
      contactExtras.join(" • ")
    ].filter(Boolean).join(" • ");
    pushFact(state.kb.contact, contactStr);

    // Horarios
    lines.filter(l=>/(horario|lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|\b\d{1,2}:\d{2}\b|\bam\b|\bpm\b)/i.test(l))
      .slice(0,10).forEach(s=>pushFact(state.kb.hours, s));

    // Precios/ofertas
    lines.filter(l=>/(?:\$|\bUSD\b|\bCOP\b|\bMXN\b|\bS\/\b|\bAR\$)|\bprecio|\bplan|\bpaquete|\btarifa|\bval(?:or|ores)/i.test(l))
      .slice(0,20).forEach(s=>pushFact(state.kb.prices, s));

    // Políticas
    lines.filter(l=>/(pol[ií]tica|t[eé]rminos|condiciones|garant[ií]a|devoluci[oó]n|reembolso)/i.test(l))
      .slice(0,20).forEach(s=>pushFact(state.kb.policies, s));

    // Productos/Servicios/Ubicaciones
    lines.filter(l=>/(producto|servicio|portafolio|cat[aá]logo|oferta)/i.test(l)).slice(0,20).forEach(s=>pushFact(state.kb.products, s));
    lines.filter(l=>/(servicio|asesor[ií]a|soporte|mantenimiento|implementaci[oó]n)/i.test(l)).slice(0,20).forEach(s=>pushFact(state.kb.services, s));
    lines.filter(l=>/(sede|oficina|tienda|local|ciudad|direcci[oó]n|ubicaci[oó]n)/i.test(l)).slice(0,20).forEach(s=>pushFact(state.kb.locations, s));

    // FAQs simples
    for (let i=0;i<lines.length-1;i++){
      const q = lines[i]; const a = lines[i+1];
      if (/\?/.test(q) && a && !/\?$/.test(a)){
        state.kb.faqs.push({ q:q.replace(/\s+/g," ").trim(), a:a.replace(/\s+/g," ").trim() });
      }
    }

    // Frases clave + keyphrases
    const sents = text.split(/(?<=[\.\!\?])\s+/).map(s=>s.trim()).filter(x=>x.length>40);
    sents.slice(0,120).forEach(s=>{
      const weight = Math.min(1.0, (tokens(s).length/25));
      state.kb.sentences.push({ text:s, weight, doc:doc.title });
      tokens(s).forEach(t=> addKey(t, 0.1));
    });
    const tf = new Map();
    tokens(text).forEach(t=> tf.set(t,(tf.get(t)||0)+1));
    Array.from(tf.entries()).sort((a,b)=> b[1]-a[1]).slice(0,40).forEach(([t,w])=> addKey(t, w*0.25));
  });

  // Compactar
  state.kb.contact = [...new Set(state.kb.contact)].filter(Boolean).slice(0,5);
  ["hours","prices","policies","products","services","locations"].forEach(k=>{
    state.kb[k] = [...new Set(state.kb[k])].filter(Boolean).slice(0,20);
  });
  state.kb.faqs = state.kb.faqs.slice(0,20);
}

/* ===================== Backend IA opcional ===================== */
function getHistory(scope, maxTurns=8){
  const arr = (scope==="mini") ? state.miniChat : state.chat;
  return arr.slice(-maxTurns).map(m=>({ role: m.role, text: m.text }));
}
async function askServerAI(q, scope, draft){
  if (!AI_SERVER_URL) return null;
  const urlSources = state.sources.filter(s => s.type==='url' && s.href).map(s => s.href);
  const useWeb = !!state.settings?.allowWeb && urlSources.length>0;
  const body = {
    message: q,
    draft,
    history: getHistory(scope, 8),
    profile: { name: state.bot.name, goal: state.bot.goal, notes: state.bot.notes },
    allowWeb: useWeb,
    webUrls: useWeb ? urlSources.slice(0,3) : [],
    strictContext: !!state.settings?.strictContext
  };
  try{
    const r = await fetch(AI_SERVER_URL, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body) });
    const json = await r.json();
    return (json.answer||json.draft||"").trim();
  }catch(e){ console.warn("AI server error:", e); return null; }
}

/* ===================== Respuestas claras (saludo + bullets + CTA) ===================== */
const STYLE = { maxBullets: 3, followup: true, cta: "¿Te paso con un asesor o prefieres que te detalle el siguiente paso aquí?" };

function shortenSentences(text, max=STYLE.maxBullets) {
  const out = [];
  const seen = new Set();
  text.split(/(?<=[\.\!\?])\s+/).forEach(s=>{
    const t = s.replace(/\s+/g," ").trim();
    if (t.length < 25) return;
    const key = t.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    out.push(t);
  });
  return out.slice(0, max);
}
function blockFromKB(theme){
  const KB = state.kb;
  if (theme==="contact" && KB.contact.length) return shortenSentences(KB.contact.join(". "));
  if (theme==="hours"   && KB.hours.length)   return shortenSentences(KB.hours.join(". "));
  if (theme==="prices"  && (KB.prices.length||KB.products.length)) {
    const lines = (KB.products.slice(0,2).concat(KB.prices)).join(". ");
    return shortenSentences(lines);
  }
  if (theme==="policies" && KB.policies.length) return shortenSentences(KB.policies.join(". "));
  if ((theme==="general"||theme==="support") && KB.faqs.length) {
    const lines = KB.faqs.slice(0,3).map(f=>`${f.q.replace(/\?+$/,"")}: ${f.a}`);
    return shortenSentences(lines.join(". "));
  }
  return [];
}
function blockFromHits(hits){
  const lines = [];
  const seen = new Set();
  hits.forEach(h=>{
    h.chunk.text.split(/(?<=[\.\!\?])\s+/).forEach(s=>{
      const t = s.replace(/\s+/g," ").trim();
      if (t.length<30) return;
      const key = t.toLowerCase();
      if (seen.has(key)) return;
      seen.add(key);
      lines.push(t);
    });
  });
  return shortenSentences(lines.join(" "));
}
function detectTheme(q){
  const s = (q||"").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g,"");
  if (/(precio|costo|tarifa|cuanto|cotiza|presupuesto|plan|paquete)/.test(s)) return "prices";
  if (/(horario|atencion|abre|cierra|dias|agenda|cita)/.test(s)) return "hours";
  if (/(contacto|whats|telefono|correo|email|direccion|ubicacion|sede)/.test(s)) return "contact";
  if (/(politica|garantia|devolucion|reembolso|termino|condicion|privacidad|tyc)/.test(s)) return "policies";
  if (/(producto|servicio|portafolio|catalogo|oferta)/.test(s)) return "products";
  if (/(soporte|ayuda|ticket|incidencia|pqrs|reclamo)/.test(s)) return "support";
  return "general";
}
function composeCrispAnswer(q, hits){
  const theme = detectTheme(q);
  // preferimos KB y complementamos con evidencias
  let bullets = blockFromKB(theme);
  if (bullets.length < STYLE.maxBullets) {
    const more = blockFromHits(hits).filter(x=> !bullets.includes(x)).slice(0, STYLE.maxBullets - bullets.length);
    bullets = bullets.concat(more);
  }
  if (!bullets.length) bullets = ["Puedo ayudarte con productos/servicios, precios, horarios, contacto y soporte usando la información cargada."];

  const title = (state.bot.name ? `¡Hola! Soy el asistente de ${state.bot.name}.` : `¡Hola!`);
  const body  = "• " + bullets.join("\n• ");
  const cta   = STYLE.cta;

  return `${title}\n\n${body}\n\n${STYLE.followup ? cta : ""}`.trim();
}

/* ===================== Export HTML estático (offline) ===================== */
function exportStandaloneHtml(){
  const payload = {
    meta: { exportedAt: new Date().toISOString(), app: "Studio Chatbot v2" },
    bot: state.bot, qa: state.qa,
    docs: state.docs.map(d => ({ title: d.title, text: d.text }))
  };
  const html = `<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>${(state.bot.name||"Asistente")} — Chat</title><style>:root{--bg:#0f1221;--text:#e7eaff;--brand:#6c8cff;--accent:#22d3ee}*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:radial-gradient(1000px 500px at 10% -10%,rgba(108,140,255,.15),transparent),radial-gradient(800px 400px at 90% -10%,rgba(34,211,238,.08),transparent),var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Helvetica Neue,Noto Sans,Arial}.wrap{max-width:900px;margin:0 auto;padding:20px}.card{border:1px solid rgba(255,255,255,.08);border-radius:16px;background:rgba(255,255,255,.04);padding:16px}.header{display:flex;gap:10px;align-items:center;margin-bottom:12px}.logo{width:28px;height:28px;border-radius:8px;background:conic-gradient(from 200deg at 60% 40%,var(--brand),var(--accent))}.title{font-weight:700}.chatlog{min-height:60vh;display:flex;flex-direction:column;gap:8px;overflow:auto;padding:6px}.bubble{max-width:80%;padding:10px 12px;border-radius:14px}.user{align-self:flex-end;background:rgba(108,140,255,.18);border:1px solid rgba(108,140,255,.45)}.bot{align-self:flex-start;background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.45)}.composer{display:flex;gap:10px;margin-top:10px}.composer input{flex:1;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:rgba(5,8,18,.6);color:var(--text)}.composer button{padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:linear-gradient(180deg,rgba(108,140,255,.25),rgba(108,140,255,.06));color:var(--text)}</style></head><body><div class="wrap"><div class="card"><div class="header"><div class="logo"></div><div><div class="title">${(state.bot.name||"Asistente")}</div><div style="opacity:.7">${state.bot.goal||""}</div></div></div><div id="log" class="chatlog"></div><div class="composer"><input id="ask" placeholder="Escribe..."/><button id="send">Enviar</button></div><div style="opacity:.7;margin-top:6px">Modo: RAG local (offline) • Híbrido+BM25</div></div></div><script>window.BOOT=${JSON.stringify(payload)};</script><script>(function(){const STOP=new Set("a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos el ella ellas ellos en entre era erais éramos eran es esa esas ese esos esta estaba estabais estábamos estaban estar este esto estos fue fui fuimos ha han hasta hay la las le les lo los mas más me mientras muy nada ni nos o os otra otros para pero poco por porque que quien se ser si sí sin sobre soy su sus te tiene tengo tuvo tuve u un una unas unos y ya".split(/\\s+/));const strip=s=>s.normalize('NFD').replace(/[\\u0300-\\u036f]/g,"");const stem=w=>strip(w).toLowerCase().replace(/(mente|ciones|cion|idades|idad|osos?|osas?|ando|iendo|ados?|idas?|es|s)$/,'');const tokens=t=>strip(t.toLowerCase()).replace(/[^a-z0-9áéíóúñü\\s]/gi,' ').split(/\\s+/).filter(w=>w&&!STOP.has(w)&&w.length>1);const chunk=(txt,sz=1200,ov=120)=>{const w=txt.split(/\\s+/);const out=[];for(let i=0;i<w.length;i+=Math.max(1,Math.floor(sz-ov))){const part=w.slice(i,i+sz).join(' ').trim();if(part.length>40) out.push(part);}return out};function build(docs){const vocab=new Map();const chs=[];let total=0;docs.forEach(d=>{chunk(d.text).forEach((t,i)=>{const toks=tokens(t).map(stem);const tf=new Map();toks.forEach(x=>tf.set(x,(tf.get(x)||0)+1));const c={id:d.title+\"#\"+i,text:t,tf,len:toks.length,vec:new Map()};const seen=new Set();toks.forEach(x=>{if(!seen.has(x)){vocab.set(x,(vocab.get(x)||0)+1);seen.add(x);} });chs.push(c);total+=c.len;});});const N=chs.length||1;const avgdl=total/N;const idf=new Map();const idfB=new Map();for(const [term,df] of vocab){idf.set(term,Math.log((N+1)/(df+1))+1);idfB.set(term,Math.log(((N-df+0.5)/(df+0.5))+1));}chs.forEach(c=>{const v=new Map();for(const [t,f] of c.tf){v.set(t,(f/Math.max(1,c.len))*(idf.get(t)||0));}c.vec=v;});return {chs,idf,idfB,N,avgdl};}function cos(a,b){let d=0,na=0,nb=0;a.forEach((va,t)=>{const vb=b.get(t)||0;d+=va*vb;na+=va*va});b.forEach(vb=>nb+=vb*vb);return (na&&nb)?(d/(Math.sqrt(na)*Math.sqrt(nb))):0}const boot=window.BOOT;const IDX=build(boot.docs);const expand=q=>{const ts=tokens(q).map(stem);const out=new Set(ts);ts.forEach(t=>{if(t.endsWith('s')) out.add(t.replace(/s$/,'')); else out.add(t+'s'); out.add(strip(t));});return Array.from(out)};function bm25(qTerms,ch){const k1=1.2,b=0.75;let s=0;for(const t of qTerms){const idf=IDX.idfB.get(t)||0;const tf=ch.tf.get(t)||0;const denom=tf + k1*(1 - b + b*(ch.len/IDX.avgdl)); if(denom>0) s+=idf*((tf*(k1+1))/denom);}return s}function hybrid(q,ch){const ex=expand(q);const tf=new Map();ex.forEach(t=>tf.set(t,(tf.get(t)||0)+1));const v=new Map();const n=Math.max(1,ex.length);ex.forEach(t=>v.set(t,(tf.get(t)/n)*(IDX.idf.get(t)||0.5)));const cosv=cos(v,ch.vec);const setQ=new Set(ex);const setC=new Set(tokens(ch.text).map(stem));let inter=0;setC.forEach(x=>{if(setQ.has(x)) inter++});const jac= (setQ.size+setC.size-inter)? inter/(setQ.size+setC.size-inter):0;return 0.7*cosv+0.3*jac}function search(q,k=6,thr=0.12){const ex=expand(q);const arr=[];let minB=Infinity,maxB=-Infinity;IDX.chs.forEach(c=>{const h=hybrid(q,c);const b=bm25(ex,c);arr.push({c,h,b});if(b<minB)minB=b;if(b>maxB)maxB=b});const range=(maxB-minB)||1;arr.forEach(o=>o.s = 0.55*o.h + 0.45*((o.b-minB)/range));arr.sort((a,b)=>b.s-a.s);const filt=arr.filter(o=>o.s>=thr).slice(0,k);return (filt.length?filt:arr.slice(0,Math.min(k,6))).map(o=>({c:o.c,s:o.s}))}const log=document.getElementById(\"log\");const push=(role,text)=>{const b=document.createElement(\"div\");b.className=\"bubble \"+(role===\"user\"?\"user\":\"bot\");b.textContent=text;log.appendChild(b);log.scrollTop=log.scrollHeight};function compose(q,hits){const sents=[];const seen=new Set();hits.forEach(x=>x.c.text.split(/(?<=[\\.\\!\\?])\\s+/).forEach(y=>{const t=y.trim();if(t.length<40)return;const k=t.toLowerCase();if(seen.has(k))return;seen.add(k);sents.push({t,sc:x.s});}));sents.sort((a,b)=>b.sc-a.sc);const pick=sents.slice(0,6).map(x=>'• '+x.t).join('\\n');return pick || \"Resumen:\\n\"+IDX.chs.slice(0,5).map(c=>'• '+c.text.slice(0,160)+'…').join('\\n');}function handle(){const i=document.getElementById(\"ask\");const q=i.value.trim();if(!q)return;i.value=\"\";push(\"user\",q);const hits=search(q,6,0.12);const bullets=compose(q,hits);const head=${JSON.stringify(state.bot.name?`¡Hola! Soy el asistente de ${state.bot.name}.`:"¡Hola!")};const cta=${JSON.stringify(STYLE.cta)};push(\"bot\", head+\"\\n\\n\"+bullets+\"\\n\\n\"+cta);}document.getElementById(\"send\").addEventListener(\"click\",handle);document.getElementById(\"ask\").addEventListener(\"keydown\",e=>{if(e.key===\"Enter\"&&!e.shiftKey){e.preventDefault();handle()}});})();</script></body></html>`;
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (state.bot.name ? state.bot.name.toLowerCase().replace(/\s+/g,"-") : "asistente") + "-static.html";
  document.body.appendChild(a); a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1500);
  a.remove();
}

/* ===================== JSONL limpio: limpieza, extracción y generación ===================== */
// LIMPIEZA AGRESIVA (quita Title/URL/Markdown/menús/cookies/links/imágenes)
function cleanForAnswer(input){
  let t = (input||"");

  // Quita URLs primero (evita falsos positivos en teléfonos)
  t = t.replace(/https?:\/\/\S+/gi, " ");

  // Encabezados de scrapers (r.jina.ai, etc.)
  t = t.replace(/^(Title|URL Source|Published Time|Markdown Content|Image \d+):.*$/gmi, " ");

  // Markdown: imágenes y links
  t = t.replace(/!\[[^\]]*\]\([^)]+\)/g, " ");
  t = t.replace(/\[[^\]]*\]\([^)]+\)/g, " ");

  // Menús/cookies/redes
  t = t.replace(/\b(inicio|home|blog|tienda|contacto|cart|carrito|mi cuenta|account)\b/gi, " ");
  t = t.replace(/\b(Accept|Decline|este sitio utiliza cookies.*|Go to [A-Za-z ]+ page)\b/gi, " ");

  // Restos HTML y entidades
  t = t.replace(/<script[\s\S]*?<\/script>/gi, " ")
       .replace(/<style[\s\S]*?<\/style>/gi, " ")
       .replace(/<[^>]+>/g, " ")
       .replace(/&[a-z]+;/gi, " ");

  // Espacios
  t = t.replace(/[ \t]+/g, " ").replace(/\s{2,}/g, " ").trim();
  return t;
}

// Frases “buenas” para respuestas (filtra más ruido)
function topSentences(text, max=5, minLen=50){
  const out = [];
  const seen = new Set();
  cleanForAnswer(text).split(/(?<=[\.\!\?])\s+/).forEach(s=>{
    const t = s.replace(/\s+/g," ").trim();
    if (t.length < minLen) return;
    const key = t.toLowerCase();
    if (seen.has(key)) return;
    if (/^(title|markdown content|url source|published time)\b/i.test(t)) return;
    if (/(\[|\]|\(|\)|\bimg\b|\bimage\b)/i.test(t)) return;
    seen.add(key);
    out.push(t);
  });
  return out.slice(0, max);
}

// Extracción robusta desde URL (emails, teléfonos “reales”, horarios, ofertas, políticas, FAQs)
function extractFromText(url, rawText){
  const text = cleanForAnswer(rawText);
  const lines = text.split(/\n+/).map(s=>s.trim()).filter(Boolean);

  // Emails
  const emails = Array.from((text||"").matchAll(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi)).map(m=>m[0]);

  // Teléfonos: 7–12 dígitos, normalizados, sin números basura
  const phoneCandidates = Array.from((text||"").matchAll(/\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b/g))
    .map(m=>m[0]);
  const phones = Array.from(new Set(phoneCandidates
    .map(p=>p.replace(/[^\d+]/g,""))
    .filter(d=>{
      const digits = d.replace(/\D/g,"");
      const len = digits.length;
      return len>=7 && len<=12;
    })
  ));

  // Horarios
  const hours = lines.filter(l=>/(horario|lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|\b\d{1,2}:\d{2}\b|\bam\b|\bpm\b)/i.test(l)).slice(0,8);

  // Ofertas/Precios
  const offers = lines.filter(l=>/(precio|plan|paquete|servicio|producto|\$|\bUSD\b|\bCOP\b|\bMXN\b)/i.test(l)).slice(0,12);

  // Políticas
  const policies = lines.filter(l=>/(pol[ií]tica|t[eé]rminos|condiciones|garant[ií]a|devoluci[oó]n|reembolso|privacidad)/i.test(l)).slice(0,12);

  // Descripción compacta
  const desc = topSentences(text, 4, 50).join(" ");

  // FAQs (línea con ? + siguiente línea como respuesta)
  const faqs = [];
  for (let i=0;i<lines.length-1;i++){
    const q = lines[i], a = lines[i+1];
    if (/\?/.test(q) && a && !/\?$/.test(a)) {
      const ans = topSentences(a, 2, 30).join(" ");
      if (ans) faqs.push({ q: q.replace(/\s+/g," ").trim(), a: ans });
    }
  }

  // Contacto legible
  const contactParts = [];
  if (emails.length) contactParts.push(`Email(s): ${[...new Set(emails)].join(", ")}`);
  if (phones.length) contactParts.push(`Teléfono(s): ${[...new Set(phones)].join(", ")}`);
  const contactLines = lines.filter(l=>/(whats?app|contacto|direcci[oó]n|ubicaci[oó]n|soporte|correo|email|tel[eé]fono|celular)/i.test(l)).slice(0,3);
  const contact = [ ...contactParts, ...contactLines ].filter(Boolean).join(" • ");

  // Nombre aproximado por dominio
  let name=""; try { const host = new URL(url).hostname.replace(/^www\./,''); name = host.split('.')[0]; } catch {}
  name = name ? (name.charAt(0).toUpperCase()+name.slice(1)) : "";

  return { name, desc, contact, hours, offers, policies, faqs };
}

// Pares canónicos (limpios) para JSONL
function pairsCanonicalFromBuckets(b){
  const bullets = (arr, n=5)=> arr.slice(0,n).map((x,i)=> `${i+1}. ${x}`).join("\n");
  const pairs = [];

  if (b.desc) {
    pairs.push({ q: "¿Qué hacen? | ¿Qué es esta empresa? | ¿A qué se dedican?", a: b.desc, tags:["about"], src:"perfil" });
  }
  if (b.offers && b.offers.length){
    pairs.push({ q: "¿Qué productos o servicios ofrecen? | ¿Cuáles son los planes y precios?", a: bullets(b.offers, 6), tags:["oferta","precios"], src:"oferta" });
  }
  if (b.contact){
    pairs.push({ q: "¿Cómo los contacto? | ¿Tienen WhatsApp o teléfono? | ¿Cuál es el correo?", a: b.contact, tags:["contacto"], src:"contacto" });
  }
  if (b.hours && b.hours.length){
    pairs.push({ q: "¿Cuáles son los horarios de atención? | ¿Abren fines de semana?", a: bullets(b.hours, 6), tags:["horarios"], src:"operación" });
  }
  if (b.policies && b.policies.length){
    pairs.push({ q: "¿Tienen políticas de garantía, cambios o devoluciones?", a: bullets(b.policies, 6), tags:["políticas"], src:"políticas" });
  }
  (b.faqs||[]).slice(0,8).forEach(f=>{
    const Q = (f.q||"").trim(); const A = (f.a||"").trim();
    if (Q && A) pairs.push({ q: Q, a: A, tags:["faq"], src:"faq" });
  });

  // Saludo robusto
  pairs.push({
    q: "hola | buenas | hey | hello | hi",
    a: `¡Hola! Soy el asistente virtual${state.bot.name?` de ${state.bot.name}`:""}. Puedo ayudarte con: productos/servicios, precios, horarios, contacto y soporte. ¿Qué necesitas?`,
    tags:["saludo"], src:"sistema"
  });

  return pairs.filter(p=> p.q && p.a);
}

// Wizard → construir pares desde campos
function pairsFromWizard(){
  const name = ($("w_name")?.value||"").trim();
  const tone = ($("w_tone")?.value||"").trim() || "Cercano y profesional.";
  const desc = ($("w_desc")?.value||"").trim();
  const contact = ($("w_contact")?.value||"").trim();
  const hours = ($("w_hours")?.value||"").trim().split(/\r?\n/).filter(Boolean);
  const offers = ($("w_offers")?.value||"").trim().split(/\r?\n/).filter(Boolean);
  const policies = ($("w_policies")?.value||"").trim().split(/\r?\n/).filter(Boolean);

  const faqs = [];
  ($("w_faqs")?.value||"").trim().split(/\r?\n/).forEach(line=>{
    const parts = line.split(/\s*\|\s*/);
    if (parts.length>=2) {
      const q = (parts[0]||"").trim();
      const a = (parts.slice(1).join(" | ")||"").trim();
      if (q && a) faqs.push({ q: /[\?\¿]$/.test(q)? q : (q+"?"), a });
    }
  });

  const buckets = { desc, contact, hours, offers, policies, faqs };
  const pairs = pairsCanonicalFromBuckets(buckets);

  if (name || desc) {
    pairs.unshift({ q: `¿Quiénes son ${name||"ustedes"}?`, a: `${desc}\n\nTono del asistente: ${tone}`, tags:["about"], src:"perfil" });
  }
  return pairs;
}
async function autoFillFromUrl(){
  const elUrl = $("wizardUrlInput"); if (!elUrl) return;
  const url = elUrl.value.trim(); if (!url) return alert("Pega una URL primero.");
  const txt = await fetchUrlText(url);
  if (!txt) return alert("No pude leer esa URL (CORS). Prueba otra o usa un proxy.");

  const b = extractFromText(url, txt);

  if ($("w_name"))    $("w_name").value = b.name || $("w_name").value;
  if ($("w_desc"))    $("w_desc").value = b.desc || $("w_desc").value;
  if ($("w_contact")) $("w_contact").value = b.contact || $("w_contact").value;
  if ($("w_hours"))   $("w_hours").value = (b.hours||[]).join("\n");
  if ($("w_offers"))  $("w_offers").value = (b.offers||[]).join("\n");
  if ($("w_policies"))$("w_policies").value = (b.policies||[]).join("\n");

  const faqLines = (b.faqs||[]).map(f=> `${f.q} | ${f.a}`).join("\n");
  if ($("w_faqs")) $("w_faqs").value = faqLines;

  alert("Campos autocompletados. Revisa y edita antes de generar el JSONL.");
}
function pairsToJSONL(pairs){
  return pairs.map(p=> JSON.stringify({
    q: p.q, a: p.a, src: p.src||undefined, tags: p.tags||undefined
  })).join("\n");
}
function previewJSONL(){
  const pairs = pairsFromWizard();
  const out = pairsToJSONL(pairs);
  if ($("jsonlPreview")) $("jsonlPreview").value = out;
}
function downloadJSONL(){
  const text = $("jsonlPreview")?.value || pairsToJSONL(pairsFromWizard());
  const blob = new Blob([text], { type:"application/jsonl;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (state.bot.name ? state.bot.name.toLowerCase().replace(/\s+/g,"-") : "dataset") + ".jsonl";
  document.body.appendChild(a); a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1500);
  a.remove();
}
function addJSONLToProject(){
  const text = $("jsonlPreview")?.value?.trim(); if (!text) return alert("No hay contenido JSONL.");
  const lines = text.split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
  const qaPairs = [];
  for (const line of lines){
    try{
      const obj = JSON.parse(line);
      if (obj && obj.q && obj.a) qaPairs.push({ q:String(obj.q), a:String(obj.a), src: obj.src?String(obj.src):undefined });
    }catch{}
  }
  if (!qaPairs.length) return alert("No encontré pares válidos {q,a} en el JSONL.");
  state.qa.push(...qaPairs);
  const txt = qaPairs.map(x=>`PREGUNTA: ${x.q}\nRESPUESTA: ${x.a}${x.src?`\nFUENTE: ${x.src}`:""}`).join("\n\n");
  const sid = nowId();
  state.sources.push({id:sid, type:'file', title:'dataset.jsonl', addedAt:Date.now()});
  state.docs.push({id:nowId(), sourceId:sid, title:'dataset.jsonl', text:txt, chunks:[]});
  buildIndex(); save(); renderSources();
  alert("Dataset agregado e indexado.");
}

/* ===================== UI: render ===================== */
function renderBasics(){
  $("botName") && ( $("botName").value = state.bot.name||"" );
  $("botGoal") && ( $("botGoal").value = state.bot.goal||"" );
  $("botNotes") && ( $("botNotes").value = state.bot.notes||"" );
  $("systemPrompt") && ( $("systemPrompt").value = state.bot.system||"" );
  $("topk") && ( $("topk").value = state.bot.topk );
  $("threshold") && ( $("threshold").value = state.bot.threshold );
  $("botNameDisplay") && ( $("botNameDisplay").textContent = state.bot.name || "(sin nombre)" );
  $("botGoalDisplay") && ( $("botGoalDisplay").textContent = state.bot.goal || "" );
  $("miniTitle") && ( $("miniTitle").textContent = state.bot.name || "Asistente" );
  $("modelStatus") && ( $("modelStatus").textContent = state.docs.length ? "Con conocimiento" : "Sin entrenar" );

  const snippet =
`<!-- Widget mínimo -->
<link rel="stylesheet" href="(tus estilos)">
<div class="launcher" id="launcher">💬</div>
<div class="mini" id="mini"> ... </div>
<script src="app.js"></script>`;
  $("embedSnippet") && ( $("embedSnippet").textContent = snippet );
}
function renderSources(){
  const list = $("sourcesList"); if (!list) return;
  list.innerHTML="";
  if (!state.sources.length){
    list.appendChild(el("div",{class:"muted small", text:"Aún no has cargado fuentes."}));
    return;
  }
  const items = state.sources.slice().sort((a,b)=> b.addedAt-a.addedAt);
  items.forEach(s=>{
    const badge = el("div",{class:"badge"});
    const meta = el("div",{},[
      el("div",{text:s.title}),
      el("div",{class:"small muted", text: s.type==='url'?(s.href||'URL') : (s.type==='meta'?'Perfil del bot':'Archivo')})
    ]);
    const open = s.href ? el("a",{href:s.href, target:"_blank", class:"small muted", text:"Ver"}) : el("span",{class:"small muted", text:""});
    const row = el("div",{class:"item"},[badge, meta, open]);
    list.appendChild(row);
  });
}
function renderCorpus(){
  const list = $("corpusList"); if (!list) return;
  list.innerHTML="";
  if (!state.docs.length){
    list.appendChild(el("div",{class:"muted small", text:"Sin documentos. Sube archivos o añade URLs."}));
    return;
  }
  state.docs.forEach(d=>{
    const lines = d.text.split(/\n/).slice(0,3).join(" ").slice(0,140);
    const row = el("div",{class:"item"},[
      el("div",{class:"badge"}),
      el("div",{},[
        el("div",{text:d.title}),
        el("div",{class:"sub", text: lines+(d.text.length>140?'…':'')})
      ]),
      el("span",{class:"small muted", text:`${(d.chunks?.length)||0} chunks`})
    ]);
    list.appendChild(row);
  });
}
function renderUrlQueue(){
  const list = $("urlList"); if (!list) return;
  list.innerHTML="";
  if (!state.urlsQueue.length){
    list.appendChild(el("div",{class:"muted small", text:"No hay URLs en cola."}));
    return;
  }
  state.urlsQueue.forEach(u=>{
    const row = el("div",{class:"item"},[
      el("div",{class:"badge"}),
      el("div",{},[
        el("div",{text:u.title||u.url}),
        el("div",{class:"sub", text:u.url})
      ]),
      el("button",{class:"ghost small", text:"Quitar"})
    ]);
    row.querySelector("button").addEventListener("click", ()=>{
      state.urlsQueue = state.urlsQueue.filter(x=> x.id!==u.id);
      save(); renderUrlQueue();
    });
    list.appendChild(row);
  });
}
function renderChat(){
  const log = $("chatlog"); if (!log) return;
  log.innerHTML="";
  state.chat.forEach(m=>{
    const b = el("div",{class:`bubble ${m.role==='user'?'user':'bot'}`});
    b.textContent = m.text; log.appendChild(b);
  });
  log.scrollTop = log.scrollHeight;
}
function renderMiniChat(){
  const log = $("miniLog"); if (!log) return;
  log.innerHTML="";
  state.miniChat.forEach(m=>{
    const b = el("div",{class:`bubble ${m.role==='user'?'user':'bot'}`});
    b.textContent = m.text; log.appendChild(b);
  });
  log.scrollTop = log.scrollHeight;
}
function setBusy(flag){
  ["btnIngestFiles","btnCrawl","btnTrain","btnRebuild","btnReset"].forEach(id=>{ if ($(id)) $(id).disabled = flag; });
}

/* ===================== Eventos ===================== */
function bindEvents(){
  $("botName")?.addEventListener("input", e=>{
    state.bot.name = e.target.value;
    $("botNameDisplay") && ( $("botNameDisplay").textContent = state.bot.name || "(sin nombre)" );
    $("miniTitle") && ( $("miniTitle").textContent = state.bot.name || "Asistente" );
    save();
  });
  $("botGoal")?.addEventListener("input", e=>{ state.bot.goal = e.target.value; save(); });
  $("botNotes")?.addEventListener("input", e=>{ state.bot.notes = e.target.value; save(); });
  $("systemPrompt")?.addEventListener("input", e=>{ state.bot.system = e.target.value; save(); });

  $("btnTrain")?.addEventListener("click", ()=>{ buildIndex(); $("modelStatus") && ( $("modelStatus").textContent="Con conocimiento"); alert("Entrenamiento (índice + BM25 + análisis) listo."); });
  $("topk")?.addEventListener("change", e=>{ state.bot.topk = Number(e.target.value)||5; save(); });
  $("threshold")?.addEventListener("change", e=>{ state.bot.threshold = Number(e.target.value)||0.15; save(); });

  $("btnIngestFiles")?.addEventListener("click", async ()=>{
    const files = $("filePicker").files;
    if (!files || !files.length) return alert("Selecciona archivos primero.");
    await ingestFiles(Array.from(files));
  });
  $("filePicker")?.addEventListener("change", async ()=>{
    if ($("autoTrain")?.checked){
      const files = $("filePicker").files;
      await ingestFiles(Array.from(files));
      $("filePicker").value = "";
    }
  });

  $("btnAddUrl")?.addEventListener("click", ()=>{
    const url = $("urlInput").value.trim(); if (!url) return;
    state.urlsQueue.push({id:nowId(), url, title:""}); $("urlInput").value = "";
    save(); renderUrlQueue();
  });
  $("btnCrawl")?.addEventListener("click", async ()=>{
    if (!state.urlsQueue.length) return alert("Añade al menos una URL.");
    await ingestUrls(state.urlsQueue); state.urlsQueue = []; save(); renderUrlQueue();
  });
  $("btnClearSources")?.addEventListener("click", ()=>{ state.urlsQueue = []; save(); renderUrlQueue(); });

  $("btnSearchCorpus")?.addEventListener("click", ()=>{
    const q = $("searchCorpus").value.trim(); if (!q) return;
    const hits = searchChunks(q, state.bot.topk, state.bot.threshold);
    const list = $("corpusList"); if (!list) return;
    list.innerHTML="";
    if (!hits.length){ list.appendChild(el("div",{class:"muted small", text:"Sin coincidencias."})); return; }
    hits.forEach(h=>{
      const row = el("div",{class:"item"},[
        el("div",{class:"badge"}),
        el("div",{},[
          el("div",{text:h.doc.title}),
          el("div",{class:"sub", text:h.chunk.text.slice(0,220)+"…"})
        ]),
        el("div",{class:"small muted", text:`score ${h.sFinal.toFixed(2)}`})
      ]);
      list.appendChild(row);
    });
  });

  $("btnRebuild")?.addEventListener("click", ()=>{ state.docs.forEach(d=> d.chunks=[]); buildIndex(); save(); alert("Reconstruido índice + análisis."); });
  $("btnReset")?.addEventListener("click", ()=>{
    if (!confirm("Esto borrará todo.")) return;
    state.sources=[]; state.docs=[]; state.index={vocab:new Map(), idf:new Map(), idfBM25:new Map(), built:false, N:0, avgdl:0};
    state.urlsQueue=[]; state.chat=[]; state.miniChat=[]; state.qa=[];
    state.kb={contact:[],hours:[],prices:[],policies:[],products:[],services:[],locations:[],faqs:[],sentences:[],keyphrases:new Map()};
    state.settings = { allowWeb:true, strictContext:true };
    $("ingestProgress") && ( $("ingestProgress").style.width="0%" );
    save(); renderSources(); renderCorpus(); renderUrlQueue(); renderChat(); renderMiniChat();
    $("modelStatus") && ( $("modelStatus").textContent = "Sin entrenar" );
  });

  $("allowWeb")?.addEventListener("change", e=>{ state.settings.allowWeb = !!e.target.checked; save(); });
  $("strictContext")?.addEventListener("change", e=>{ state.settings.strictContext = !!e.target.checked; save(); });
  $("btnExportHtml")?.addEventListener("click", exportStandaloneHtml);

  $("send")?.addEventListener("click", ()=> handleAsk("ask","tester"));
  $("ask")?.addEventListener("keydown", (e)=>{ if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); handleAsk("ask","tester"); } });
  $("launcher")?.addEventListener("click", ()=>{ $("mini")?.classList.add("show"); });
  $("closeMini")?.addEventListener("click", ()=>{ $("mini")?.classList.remove("show"); });
  $("miniSend")?.addEventListener("click", ()=> handleAsk("miniAsk","mini"));
  $("miniAsk")?.addEventListener("keydown", (e)=>{ if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); handleAsk("miniAsk","mini"); } });

  bindWizardEvents();
}

/* ===================== Wizard JSONL: binds ===================== */
function bindWizardEvents(){
  $("btnAutoFillFromUrl")?.addEventListener("click", autoFillFromUrl);
  $("btnPreviewJSONL")?.addEventListener("click", previewJSONL);
  $("btnDownloadJSONL")?.addEventListener("click", downloadJSONL);
  $("btnAddJSONLToProject")?.addEventListener("click", addJSONLToProject);
}

/* ===================== Chat handling ===================== */
function pushAndRender(scope, role, text){
  const arr = (scope==="mini") ? state.miniChat : state.chat;
  arr.push({role, text});
  (scope==="mini") ? renderMiniChat() : renderChat();
  save();
}
function answerFromQA(query){
  if (!state.qa.length) return null;
  const qv = vectorizeQuery(query);
  let best=null, bestScore=0;
  for (let i=0;i<state.qa.length;i++){
    const s = cosineSim(qv, vectorizeQuery(state.qa[i].q));
    if (s>bestScore){ bestScore=s; best = state.qa[i]; }
  }
  return (bestScore>=0.30) ? best : null;
}
function handleAsk(inputId, scope){
  const input = $(inputId); if (!input) return;
  const q = input.value.trim(); if (!q) return;
  input.value = "";

  const qLower = q.toLowerCase();

  // Saludos → respuesta clara con bullets
  const greetRE = /^(hola|buenas|hey|hello|hi)\b/i;
  pushAndRender(scope, 'user', q);
  if (greetRE.test(qLower)){
    const hits = searchChunks("presentación empresa servicios contacto precios horarios", state.bot.topk, Math.min(0.12,state.bot.threshold));
    const msg = composeCrispAnswer(q, hits);
    pushAndRender(scope,'assistant', msg);
    return;
  }

  // Preguntas explícitas sobre IA
  if (/(eres|tú eres|tu eres).*(ia|inteligencia|modelo|chatgpt|gemini)/i.test(qLower)){
    pushAndRender(scope,'assistant', `Soy tu asistente virtual. ¿En qué puedo ayudarte hoy?`);
    return;
  }

  // 1) Q&A programadas primero
  const qa = answerFromQA(q);
  if (qa){
    pushAndRender(scope, 'assistant', qa.a + (qa.src ? `\n\nFuente: ${qa.src}` : ""));
    return;
  }

  // 2) RAG híbrido + BM25
  const hits = searchChunks(q, state.bot.topk, state.bot.threshold);
  let draft = composeCrispAnswer(q, hits);

  // 3) Opcional: IA backend para pulir
  if (AI_SERVER_URL){
    askServerAI(q, scope, draft).then(ai=>{
      pushAndRender(scope,'assistant', (ai||draft));
    });
    return;
  }

  // 4) Respuesta clara local
  pushAndRender(scope,'assistant', draft);
}

/* ===================== Init ===================== */
(function init(){
  load();
  renderBasics();
  renderSources();
  renderCorpus();
  renderUrlQueue();
  renderChat();
  renderMiniChat();
  bindEvents();
  if (state.docs.length && !state.index.built) buildIndex();
})();
