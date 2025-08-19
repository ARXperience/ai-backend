/* app.js – Studio Chatbot v2 (GENÉRICO con Wizard JSONL FIXED)
   - RAG TF-IDF + coseno (archivos/URLs + meta: objetivo/notas/sistema)
   - Q&A via *.jsonl ({q,a,src})
   - Fallback opcional a backend IA (historial + contexto)
   - Asistente de dataset (JSONL) + Exportar HTML estático
*/

/* ======= Config backend IA (opcional). Deja "" si no tienes ======= */
const AI_SERVER_URL = ""; // ej: "https://api.tu-dominio.com/chat"

/* ===================== Estado global ===================== */
const state = {
  bot: { name:"", goal:"", notes:"", system:"", topk:5, threshold:0.15 },
  sources: /** @type {Array<Source>} */ ([]),
  docs:    /** @type {Array<Doc>} */   ([]),   // incluye meta-doc
  index:   { vocab:new Map(), idf:new Map(), built:false },
  urlsQueue: [],
  chat: [],
  miniChat: [],
  qa: /** @type {Array<{q:string,a:string,src?:string}>} */([]),
  settings: { allowWeb: true, strictContext: true }
};
let ingestBusy = false;

/* ===================== Tipos (JSDoc) ===================== */
/** @typedef {{id:string, type:'file'|'url'|'meta', title:string, href?:string, addedAt:number}} Source */
/** @typedef {{id:string, sourceId:string, title:string, text:string, chunks:Array<Chunk>, meta?:boolean}} Doc */
/** @typedef {{id:string, text:string, vector:Map<string, number>}} Chunk */

/* ===================== Helpers ===================== */
const $ = (id) => document.getElementById(id);
const el = (tag, attrs={}, children=[])=>{
  const n = document.createElement(tag);
  Object.entries(attrs).forEach(([k,v])=>{
    if (k==='class') n.className = v; else if (k==='text') n.textContent = v; else n.setAttribute(k,v);
  });
  children.forEach(c => n.appendChild(c));
  return n;
};
const nowId = ()=> Math.random().toString(36).slice(2)+Date.now().toString(36);

/* ===================== Persistencia ===================== */
const STORAGE_KEY = "studio-chatbot-v2";
function save() {
  const toSave = {
    bot: state.bot,
    sources: state.sources,
    docs: state.docs.map(d=>({ id:d.id, sourceId:d.sourceId, title:d.title, text:d.text, meta: !!d.meta })),
    urlsQueue: state.urlsQueue,
    qa: state.qa,
    chat: state.chat,
    miniChat: state.miniChat,
    settings: state.settings
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
    state.docs = (data.docs||[]).map(d=>({...d, chunks:[], meta: !!d.meta}));
    state.urlsQueue = data.urlsQueue||[];
    state.qa = data.qa||[];
    state.chat = data.chat || [];
    state.miniChat = data.miniChat || [];
    state.settings = Object.assign({ allowWeb:true, strictContext:true }, data.settings||{});
  } catch(e){ console.warn("No se pudo cargar estado:", e); }
}

/* ===================== Tokenizador y TF-IDF ===================== */
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
  return text.toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g,"")
    .replace(/[^a-z0-9áéíóúñü\s]/gi, ' ')
    .split(/\s+/)
    .filter(w => w && !STOP.has(w) && w.length>1);
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

/* ======== “Meta-doc”: Objetivo/Notas/Sistema influye la búsqueda ======== */
function upsertMetaDoc(){
  // Elimina meta-doc anterior
  state.docs = state.docs.filter(d=> !d.meta);
  state.sources = state.sources.filter(s=> s.type!=='meta');

  const pieces = [];
  if (state.bot.goal) pieces.push(`OBJETIVO DEL BOT:\n${state.bot.goal}`);
  if (state.bot.notes) pieces.push(`NOTAS INTERNAS:\n${state.bot.notes}`);
  if (state.bot.system) pieces.push(`REGLAS DEL SISTEMA:\n${state.bot.system}`);
  if (!pieces.length) return;

  const text = pieces.join("\n\n");
  const sid = nowId();
  state.sources.push({ id:sid, type:'meta', title:'Perfil del bot', addedAt:Date.now() });
  state.docs.push({ id:nowId(), sourceId:sid, title:'Perfil del bot', text, meta:true, chunks:[] });
}

/* ===================== Índice ===================== */
function buildIndex(){
  // asegura incluir meta-doc
  upsertMetaDoc();

  const vocab = new Map();
  const all = []; // {id,text,vector,_doc}
  state.docs.forEach(doc=>{
    if (!doc.chunks?.length) {
      const cks = chunkText(doc.text);
      doc.chunks = cks.map((t,i)=>({id:`${doc.id}#${i}`, text:t, vector:new Map()}));
    }
    doc.chunks.forEach(ch=>{
      all.push({ ...ch, _doc:doc });
      const seen = new Set();
      tokens(ch.text).forEach(tok=>{
        if (!seen.has(tok)){
          vocab.set(tok, (vocab.get(tok)||0)+1);
          seen.add(tok);
        }
      });
    });
  });

  const N = all.length || 1;
  const idf = new Map();
  for (const [term, df] of vocab){
    idf.set(term, Math.log((N+1)/(df+1))+1);
  }

  all.forEach(obj=>{
    const ch = obj;
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

  // Re-ensambla chunks dentro de cada doc
  state.docs.forEach(doc=>{
    doc.chunks = all.filter(x=> x._doc.id===doc.id)
      .map(x=>({ id:x.id, text:x.text, vector:x.vector }));
  });

  state.index.vocab = vocab;
  state.index.idf = idf;
  state.index.built = true;
  renderCorpus();
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
function cosineSim(a,b){
  let dot=0, na=0, nb=0;
  a.forEach((va, t)=>{ const vb=b.get(t)||0; dot += va*vb; na += va*va; });
  b.forEach(vb=>{ nb += vb*vb; });
  if (na===0 || nb===0) return 0;
  return dot / (Math.sqrt(na)*Math.sqrt(nb));
}
function searchChunks(query, k=3, thr=0.30){
  if (!state.index.built) buildIndex();
  const qv = vectorizeQuery(query);
  const scored = [];
  state.docs.forEach(doc=>{
    doc.chunks.forEach(ch=>{
      const s = cosineSim(qv, ch.vector);
      if (s>=thr) scored.push({chunk:ch, score:s, doc});
    });
  });
  scored.sort((a,b)=> b.score - a.score);
  return scored.slice(0, k);
}

/* ===================== Q&A JSONL ===================== */
function simQ(a,b){ return cosineSim(vectorizeQuery(a), vectorizeQuery(b)); }
function answerFromQA(query){
  if (!state.qa.length) return null;
  let best = {i:-1, score:0};
  for (let i=0;i<state.qa.length;i++){
    const s = simQ(query, state.qa[i].q);
    if (s > best.score) best = {i, score:s};
  }
  return (best.score >= 0.30) ? state.qa[best.i] : null;
}

/* ===================== Fallback siempre-responde ===================== */
function genericFallback(q){
  const s = (q||"").toLowerCase();
  if (/(precio|costo|cu[aá]nto|tarifa|plan)/.test(s))
    return "Puedo estimar precios/planes si me dices qué producto/servicio, cantidad y condiciones. Puedo generar una cotización base y el siguiente paso para confirmarla.";
  if (/(horario|hora|agenda|cita)/.test(s))
    return "Puedo proponerte horarios y agendar. Dime tu zona horaria y preferencia (mañana/tarde) para darte opciones.";
  if (/(contacto|tel[eé]fono|whats|correo|direcci[oó]n)/.test(s))
    return "¿Prefieres que te contacte un agente por correo o WhatsApp? Si me dejas nombre y contacto, creo el ticket y te confirmo.";
  if (/(env[ií]o|entrega|shipping|tracking)/.test(s))
    return "Te ayudo con envíos y seguimiento. Dime número de pedido o ciudad para estimar tiempos.";
  if (/(garant[ií]a|devoluci[oó]n|reembolso|cambio|soporte)/.test(s))
    return "Puedo iniciar un caso de soporte/devolución. Indícame el problema, fecha de compra y evidencia (si aplica).";
  return "Te ayudo con información, precios, soporte y más. Cuéntame tu objetivo y datos mínimos para darte una respuesta útil.";
}

/* ===================== Lectores/Fetch ===================== */
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
      alert("Para leer PDF local, incluye pdfjs (pdfjsLib) o convierte a .txt/.md.");
      return "";
    }
  }
  alert(`Formato no soportado: .${ext}. Convierte a .txt/.md/.pdf.`);
  return "";
}

// Fetch con anti-CORS: normal → r.jina.ai (readability) → allorigins
async function fetchUrlText(url){
  // 1) Intento directo (si el sitio permite CORS)
  try{
    const r = await fetch(url, { mode:'cors' });
    const ct = (r.headers.get('content-type')||"").toLowerCase();
    const raw = await r.text();
    if (ct.includes("html")) return normalizeText(raw);
    return raw;
  }catch{}

  // 2) Readability extractor público (texto limpio)
  try{
    const cleanURL = url.replace(/^https?:\/\//,'');
    const r = await fetch(`https://r.jina.ai/http://${cleanURL}`);
    const raw = await r.text();
    if (raw && raw.length>50) return normalizeText(raw);
  }catch{}

  // 3) allorigins (proxy simple)
  try{
    const r = await fetch(`https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`);
    const raw = await r.text();
    return normalizeText(raw);
  }catch(e){
    console.warn("No se pudo rastrear URL:", url, e);
    return "";
  }
}

/* ===================== Ingesta ===================== */
async function ingestFiles(files){
  if (!files?.length) return;
  setBusy(true);
  const bar = $("ingestProgress");
  bar.style.width = "0%";
  let done = 0;

  for (const f of files){
    const ext = (f.name.split('.').pop()||"").toLowerCase();

    // JSONL Q&A
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
      done++; bar.style.width = `${Math.round(done/files.length*100)}%`; continue;
    }

    const text = await readFileAsText(f);
    if (!text) { done++; bar.style.width = `${Math.round(done/files.length*100)}%`; continue; }

    const sourceId = nowId();
    state.sources.push({id:sourceId, type:'file', title:f.name, addedAt:Date.now()});
    state.docs.push({id:nowId(), sourceId:sourceId, title:f.name, text, chunks:[]});
    done++; bar.style.width = `${Math.round(done/files.length*100)}%`;
  }

  buildIndex(); save(); renderSources(); setBusy(false);
  $("modelStatus").textContent = "Con conocimiento";
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
  $("modelStatus").textContent = "Con conocimiento";
}

/* ===================== Cliente backend IA (opcional) ===================== */
function getHistory(scope, maxTurns=8){
  const arr = (scope==="mini") ? state.miniChat : state.chat;
  return arr.slice(-maxTurns).map(m=>({ role: m.role, text: m.text }));
}
async function askServerAI(q, scope){
  if (!AI_SERVER_URL) return null;
  const lowHits = searchChunks(q, 5, 0.12);
  const ctx = lowHits.map(h => h.chunk.text.slice(0, 1600));
  const titles = lowHits.map(h => h.doc.title);
  const urlSources = state.sources.filter(s => s.type==='url' && s.href).map(s => s.href);
  const useWeb = !!state.settings?.allowWeb && urlSources.length>0;

  const body = {
    message: q,
    context: ctx,
    titles,
    history: getHistory(scope, 8),
    profile: { name: state.bot.name, goal: state.bot.goal, notes: state.bot.notes },
    allowWeb: useWeb,
    webUrls: useWeb ? urlSources.slice(0,3) : [],
    strictContext: !!state.settings?.strictContext
  };

  try{
    const r = await fetch(AI_SERVER_URL, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body) });
    const json = await r.json();
    return json.answer?.trim();
  }catch(e){ console.warn("AI server error:", e); return null; }
}

/* ===================== Síntesis (omite meta-doc en bullets) ===================== */
function synthesizeAnswer(query, hits){
  const sentences = [];
  const seen = new Set();
  hits.forEach(h=>{
    if (h.doc.meta) return; // no citar perfil/objetivo/notas
    h.chunk.text.split(/(?<=[\.\!\?])\s+/).forEach(s=>{
      const t = s.trim();
      if (!t) return;
      const key = t.toLowerCase();
      if (seen.has(key)) return;
      if (t.length<30) return;
      seen.add(key);
      sentences.push({text:t, score:h.score});
    });
  });

  sentences.sort((a,b)=> b.score - a.score);
  const picked = sentences.slice(0, 5).map(s=> s.text);
  if (!picked.length) return "";

  const bullets = picked.map(s=> "• " + s);
  const first = picked[0] || "";
  let extra = picked.find(s=> s!==first) || "";
  if (extra.length > 180) extra = extra.slice(0,180)+"…";
  const srcTitles = Array.from(new Set(hits.filter(h=>!h.doc.meta).map(h=> h.doc.title))).slice(0,3);

  return [
    `Sobre “${query.slice(0,120)}”: ${first} ${extra}`,
    bullets.join("\n"),
    srcTitles.length ? `Fuentes: ${srcTitles.join(" • ")}` : ""
  ].filter(Boolean).join("\n\n");
}
function stylizeAnswer(text, system, notes){
  let t = text;
  if (notes && /breve|conciso/i.test(notes) && t.length>600) t = t.slice(0, 600) + "…";
  if (system) t = t.replaceAll(system, ""); // evita “filtrarlo”
  return t;
}

/* ===================== Exportar HTML estático ===================== */
function exportStandaloneHtml() {
  const payload = {
    meta: { exportedAt: new Date().toISOString(), app: "Studio Chatbot v2" },
    bot: state.bot,
    qa: state.qa,
    docs: state.docs.map(d => ({ title: d.title, text: d.text })), // incluye meta
  };
  const html = `<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>${(state.bot.name||"Asistente")} — Chat</title><style>:root{--bg:#0f1221;--text:#e7eaff;--brand:#6c8cff;--accent:#22d3ee}*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:radial-gradient(1000px 500px at 10% -10%,rgba(108,140,255,.15),transparent),radial-gradient(800px 400px at 90% -10%,rgba(34,211,238,.08),transparent),var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Helvetica Neue,Noto Sans,Arial}.wrap{max-width:900px;margin:0 auto;padding:20px}.card{border:1px solid rgba(255,255,255,.08);border-radius:16px;background:rgba(255,255,255,.04);padding:16px}.header{display:flex;gap:10px;align-items:center;margin-bottom:12px}.logo{width:28px;height:28px;border-radius:8px;background:conic-gradient(from 200deg at 60% 40%,var(--brand),var(--accent))}.title{font-weight:700}.chatlog{min-height:60vh;display:flex;flex-direction:column;gap:8px;overflow:auto;padding:6px}.bubble{max-width:80%;padding:10px 12px;border-radius:14px}.user{align-self:flex-end;background:rgba(108,140,255,.18);border:1px solid rgba(108,140,255,.45)}.bot{align-self:flex-start;background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.45)}.composer{display:flex;gap:10px;margin-top:10px}.composer input{flex:1;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:rgba(5,8,18,.6);color:var(--text)}.composer button{padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:linear-gradient(180deg,rgba(108,140,255,.25),rgba(108,140,255,.06));color:var(--text)}</style></head><body><div class="wrap"><div class="card"><div class="header"><div class="logo"></div><div><div class="title">${(state.bot.name||"Asistente")}</div><div style="opacity:.7">${state.bot.goal||""}</div></div></div><div id="log" class="chatlog"></div><div class="composer"><input id="ask" placeholder="Escribe..."/><button id="send">Enviar</button></div><div style="opacity:.7;margin-top:6px">Modo: RAG local (offline) • Responder siempre</div></div></div><script>window.BOOT=${JSON.stringify(payload)};</script><script>(function(){const STOP=new Set("a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos el ella ellas ellos en entre era erais éramos eran es esa esas ese esos esta estaba estabais estábamos estaban estar este esto estos fue fui fuimos ha han hasta hay la las le les lo los mas más me mientras muy nada ni nos o os otra otros para pero poco por porque que quien se ser si sí sin sobre soy su sus te tiene tengo tuvo tuve u un una unas unos y ya".split(/\\s+/));const tokens=t=>t.toLowerCase().normalize('NFD').replace(/[\\u0300-\\u036f]/g,"").replace(/[^a-z0-9áéíóúñü\\s]/gi,' ').split(/\\s+/).filter(w=>w&&!STOP.has(w)&&w.length>1);const chunk=(txt,sz=1200,ov=120)=>{const w=txt.split(/\\s+/);const out=[];for(let i=0;i<w.length;i+=Math.max(1,Math.floor(sz-ov))){const part=w.slice(i,i+sz).join(' ').trim();if(part.length>40) out.push(part);}return out};const vec=(idf,t)=>{const tf=new Map();const toks=tokens(t);toks.forEach(x=>tf.set(x,(tf.get(x)||0)+1));const v=new Map();toks.forEach(x=>v.set(x,(tf.get(x)/toks.length)*(idf.get(x)||0)));return v};const cos=(a,b)=>{let d=0,na=0,nb=0;a.forEach((va,t)=>{const vb=b.get(t)||0;d+=va*vb;na+=va*va});b.forEach(vb=>nb+=vb*vb);return (na&&nb)?(d/(Math.sqrt(na)*Math.sqrt(nb))):0};const state={bot:BOOT.bot,qa:BOOT.qa||[],docs:[],chunks:[],idf:new Map(),built:false};function build(){const chunks=[];BOOT.docs.forEach(d=>{chunk(d.text).map((t,i)=>chunks.push({text:t,title:d.title,id:d.title+"#"+i,meta:(d.title==="Perfil del bot")}))});const vocab=new Map();chunks.forEach(ch=>{const seen=new Set();tokens(ch.text).forEach(tok=>{if(!seen.has(tok)){vocab.set(tok,(vocab.get(tok)||0)+1);seen.add(tok);}})});const N=chunks.length||1;const idf=new Map();for(const [term,df] of vocab) idf.set(term,Math.log((N+1)/(df+1))+1);chunks.forEach(ch=>{const tf=new Map();const toks=tokens(ch.text);toks.forEach(t=>tf.set(t,(tf.get(t)||0)+1));ch.vec=new Map();for(const [t,f] of tf){ch.vec.set(t,(f/toks.length)*(idf.get(t)||0));}});state.chunks=chunks;state.idf=idf;state.built=true}function search(q,k=5,thr=0.15){if(!state.built) build();const qv=vec(state.idf,q);const scored=[];state.chunks.forEach(ch=>{const s=cos(qv,ch.vec);if(s>=thr) scored.push({s,ch})});scored.sort((a,b)=>b.s-a.s);return scored.slice(0,k)}function qa(q){let best={i:-1,score:0};for(let i=0;i<state.qa.length;i++){const s=cos(vec(state.idf,q),vec(state.idf,state.qa[i].q||""));if(s>best.score) best={i,score:s}}return (best.score>=0.30)?state.qa[best.i]:null}function fallback(q){const s=q.toLowerCase();if(/(precio|costo|cu[aá]nto|tarifa|plan)/.test(s))return"Comparto precios/planes si me dices producto/servicio, cantidad y condiciones.";if(/(horario|hora|agenda|cita)/.test(s))return"Puedo proponerte horarios y agendar.";if(/(contacto|tel[eé]fono|whats|correo|direcci[oó]n)/.test(s))return"¿Prefieres contacto por correo o WhatsApp?";return"Te ayudo con información, precios y soporte. Cuéntame tu objetivo."; }function synth(q,hits){if(!hits.length) return "";const s=[];const seen=new Set();hits.forEach(h=>{if(h.ch.meta) return;h.ch.text.split(/(?<=[\\.\\!\\?])\\s+/).forEach(x=>{const t=x.trim();if(!t||t.length<30)return;const k=t.toLowerCase();if(seen.has(k))return;seen.add(k);s.push({t,sc:h.s})});});s.sort((a,b)=>b.sc-a.sc);const top=s.slice(0,5).map(x=>x.t);const first=top[0]||"";const extra=(top.find(x=>x!==first)||"").slice(0,180);const bullets=top.map(x=>"• "+x).join("\\n");const src=[...new Set(hits.filter(h=>!h.ch.meta).map(h=>h.ch.title))].slice(0,3);return \`Sobre “\${q.slice(0,120)}”: \${first} \${extra}\\n\\n\${bullets}\\n\\n\${src.length?("Fuentes: "+src.join(" • ")):""}\`}const log=document.getElementById("log");const push=(role,text)=>{const b=document.createElement("div");b.className="bubble "+(role==="user"?"user":"bot");b.textContent=text;log.appendChild(b);log.scrollTop=log.scrollHeight};function handle(){const i=document.getElementById("ask");const q=i.value.trim();if(!q)return;i.value="";push("user",q);const m=qa(q);if(m){push("bot",m.a+(m.src?("\\n\\nFuente: "+m.src):""));return}const hits=search(q,(BOOT.bot?.topk||5),(BOOT.bot?.threshold||0.15));if(!hits.length){push("bot",fallback(q));return}push("bot",synth(q,hits)||fallback(q))}document.getElementById("send").addEventListener("click",handle);document.getElementById("ask").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();handle()}});build();})();</script></body></html>`;
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (state.bot.name ? state.bot.name.toLowerCase().replace(/\s+/g,"-") : "asistente") + "-static.html";
  document.body.appendChild(a); a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1500);
  a.remove();
}

/* ===================== Wizard JSONL (FIXED) ===================== */

/** Utilidad: mensajes */
function msg(txt){ alert(txt); }

/** Parser robusto para FAQs: "¿Pregunta?|Respuesta", "Pregunta? | Respuesta", "Pregunta | Respuesta" */
function parseFaqLine(line){
  const raw = line.trim();
  if (!raw) return null;
  const parts = raw.split(/\s*\|\s*/);
  if (parts.length < 2) return null;

  let q = (parts[0]||"").trim();
  let a = (parts.slice(1).join(" | ")||"").trim();

  if (!q || !a) return null;

  const hadQMark = /[¿?]/.test(q);
  q = q.replace(/\?+$/,'');
  if (!/\?$/.test(q)) q = q + (hadQMark ? "" : "?");

  return { q, a, src:"faq" };
}

/** Genera pares Q&A desde el wizard */
function pairsFromWizard(){
  const name = ($("w_name")?.value||"").trim();
  const tone = ($("w_tone")?.value||"").trim() || "Cercano y profesional.";
  const desc = ($("w_desc")?.value||"").trim();
  const contact = ($("w_contact")?.value||"").trim();
  const hours = ($("w_hours")?.value||"").trim();
  const offersLines = ($("w_offers")?.value||"").trim().split(/\r?\n/).filter(Boolean);
  const policies = ($("w_policies")?.value||"").trim();
  const faqsLines = ($("w_faqs")?.value||"").trim().split(/\r?\n/).filter(Boolean);

  /** @type {Array<{q:string,a:string,src?:string}>} */
  const pairs = [];

  if (name || desc){
    pairs.push({
      q: `¿Qué es ${name || "esta empresa"} y qué hace?`,
      a: `${desc || "Somos una empresa que ayuda a clientes con productos y servicios específicos."}\n\nTono del asistente: ${tone}`,
      src: "perfil"
    });
  } else {
    pairs.push({
      q: "¿Qué hace esta empresa?",
      a: "Ayudamos a clientes con sus necesidades principales. Puedo orientarte en precios, procesos y soporte.",
      src: "perfil"
    });
  }

  if (contact) pairs.push({ q:"¿Cómo puedo contactarlos?", a:contact, src:"contacto" });
  if (hours)   pairs.push({ q:"¿Cuáles son los horarios y cobertura?", a:hours, src:"operación" });

  if (offersLines.length){
    const list = offersLines.map((l,i)=>`${i+1}. ${l}`).join("\n");
    pairs.push({ q:"¿Qué productos/servicios ofrecen y precios?", a:list, src:"oferta" });
  }

  if (policies) pairs.push({ q:"¿Cuáles son las políticas de garantía, cambios y tiempos?", a:policies, src:"políticas" });

  faqsLines.forEach(line=>{
    const parsed = parseFaqLine(line);
    if (parsed) pairs.push(parsed);
  });

  if (!pairs.length){
    pairs.push(...pairsFromProfile());
  }

  return pairs;
}

/** Pares rápidos desde Objetivo/Notas */
function pairsFromProfile(){
  const goal = (state.bot.goal||"").trim();
  const notes = (state.bot.notes||"").trim();
  /** @type {Array<{q:string,a:string,src?:string}>} */
  const pairs = [];
  if (goal) pairs.push({ q:"¿Cuál es el objetivo de este asistente?", a:goal, src:"perfil" });
  if (notes) pairs.push({ q:"¿Qué debo saber para atender bien al cliente?", a:notes, src:"perfil" });
  if (!goal && !notes){
    pairs.push({ q:"¿Cómo respondes?", a:"De forma clara, breve y útil. Pido datos mínimos y doy siguientes pasos concretos.", src:"perfil" });
  }
  return pairs;
}

/** Serializa Q&A a JSONL */
function jsonlString(pairs){
  return pairs.map(p=>JSON.stringify(p)).join("\n");
}

/** Descarga .jsonl */
function downloadJsonl(pairs, name="dataset_chatbot.jsonl"){
  if (!pairs.length){ msg("No hay pares para descargar."); return; }
  const blob = new Blob([jsonlString(pairs)], {type:"application/jsonl;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1200);
  a.remove();
}

/** Agrega Q&A al proyecto (y re-indexa) */
function addPairsToProject(pairs){
  if (!pairs.length){ msg("No hay pares para agregar."); return; }
  state.qa.push(...pairs);
  // también como texto indexable
  const txt = pairs.map(x=>`PREGUNTA: ${x.q}\nRESPUESTA: ${x.a}${x.src?`\nFUENTE: ${x.src}`:""}`).join("\n\n");
  const sid = nowId();
  state.sources.push({id:sid, type:'file', title:'dataset_chatbot (wizard).jsonl', addedAt:Date.now()});
  state.docs.push({id:nowId(), sourceId:sid, title:'dataset_chatbot (wizard).jsonl', text:txt, chunks:[]});
  buildIndex(); save(); renderSources();
  $("modelStatus").textContent = "Con conocimiento";
  msg(`Se agregaron ${pairs.length} pares al proyecto.`);
}

/** Lee JSONL desde la vista previa (validado) */
function readPairsFromPreview(){
  const raw = ($("w_preview")?.value||"").trim();
  if (!raw) return [];
  const lines = raw.split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
  const pairs = [];
  for (let i=0;i<lines.length;i++){
    try{
      const obj = JSON.parse(lines[i]);
      if (obj && obj.q && obj.a){ pairs.push({ q:String(obj.q), a:String(obj.a), src: obj.src?String(obj.src):undefined }); }
      else { throw new Error("Falta q/a"); }
    }catch(e){
      throw new Error(`Línea ${i+1} inválida: ${e.message}`);
    }
  }
  return pairs;
}

/* ====== Enlaces de UI del Wizard ====== */
function bindWizardEvents(){
  const openBtn = $("btnDatasetWizard");
  const closeBtn = $("wizClose");
  const modal = $("datasetModal");
  const btnPreview = $("wizPreview");
  const btnDownload = $("wizDownload");
  const btnAdd = $("wizAdd");
  const btnFromProfile = $("wizFromProfile");
  const previewArea = $("w_preview");

  if (!openBtn || !modal) return;

  openBtn.addEventListener("click", ()=> modal.classList.add("show"));
  closeBtn?.addEventListener("click", ()=> modal.classList.remove("show"));

  btnPreview?.addEventListener("click", ()=>{
    const pairs = pairsFromWizard();
    previewArea.value = jsonlString(pairs);
    msg(`Generado: ${pairs.length} pares (vista previa actualizada).`);
  });

  btnDownload?.addEventListener("click", ()=>{
    try{
      const pairs = previewArea.value.trim() ? readPairsFromPreview() : pairsFromWizard();
      downloadJsonl(pairs, "dataset_chatbot.jsonl");
    }catch(e){ msg(e.message || "No se pudo descargar."); }
  });

  btnAdd?.addEventListener("click", ()=>{
    try{
      const pairs = previewArea.value.trim() ? readPairsFromPreview() : pairsFromWizard();
      if (!pairs.length){ msg("No hay pares para agregar."); return; }
      addPairsToProject(pairs);
    }catch(e){ msg(e.message || "No se pudo agregar al proyecto."); }
  });

  btnFromProfile?.addEventListener("click", ()=>{
    const pairs = pairsFromProfile();
    previewArea.value = jsonlString(pairs);
    msg(`Generado desde Objetivo/Notas: ${pairs.length} pares.`);
  });

  // Cierra modal al pulsar fuera
  modal.addEventListener("click", (e)=>{
    if (e.target === modal) modal.classList.remove("show");
  });
}

/* ===================== UI render ===================== */
function renderBasics(){
  $("botName").value = state.bot.name||"";
  $("botGoal").value = state.bot.goal||"";
  $("botNotes").value = state.bot.notes||"";
  $("systemPrompt").value = state.bot.system||"";
  $("topk").value = state.bot.topk;
  $("threshold").value = state.bot.threshold;
  $("botNameDisplay").textContent = state.bot.name || "(sin nombre)";
  $("botGoalDisplay").textContent = state.bot.goal || "";
  $("miniTitle").textContent = state.bot.name || "Asistente";
  $("modelStatus").textContent = state.docs.length ? "Con conocimiento" : "Sin entrenar";

  if ($("allowWeb")) $("allowWeb").checked = !!state.settings.allowWeb;
  if ($("strictContext")) $("strictContext").checked = !!state.settings.strictContext;

  const snippet =
`<!-- Widget mínimo -->
<link rel="stylesheet" href="(tus estilos)">
<div class="launcher" id="launcher">💬</div>
<div class="mini" id="mini"> ... </div>
<script src="app.js"></script>`;
  $("embedSnippet").textContent = snippet;
}
function renderSources(){
  const list = $("sourcesList");
  list.innerHTML = "";
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
  const list = $("corpusList");
  list.innerHTML = "";
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
  const list = $("urlList");
  list.innerHTML = "";
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
  const log = $("chatlog");
  log.innerHTML = "";
  state.chat.forEach(m=>{
    const b = el("div",{class:`bubble ${m.role==='user'?'user':'bot'}`});
    b.textContent = m.text;
    log.appendChild(b);
  });
  log.scrollTop = log.scrollHeight;
}
function renderMiniChat(){
  const log = $("miniLog");
  log.innerHTML = "";
  state.miniChat.forEach(m=>{
    const b = el("div",{class:`bubble ${m.role==='user'?'user':'bot'}`});
    b.textContent = m.text;
    log.appendChild(b);
  });
  log.scrollTop = log.scrollHeight;
}
function setBusy(flag){
  ingestBusy = flag;
  ["btnIngestFiles","btnCrawl","btnTrain","btnRebuild","btnReset"].forEach(id=>{
    if ($(id)) $(id).disabled = flag;
  });
}

/* ===================== Eventos ===================== */
function bindEvents(){
  $("botName").addEventListener("input", e=>{
    state.bot.name = e.target.value;
    $("botNameDisplay").textContent = state.bot.name || "(sin nombre)";
    $("miniTitle").textContent = state.bot.name || "Asistente";
    save();
  });
  $("botGoal").addEventListener("input", e=>{ state.bot.goal = e.target.value; save(); });
  $("botNotes").addEventListener("input", e=>{ state.bot.notes = e.target.value; save(); });
  $("systemPrompt").addEventListener("input", e=>{ state.bot.system = e.target.value; save(); });

  // Entrenar / parámetros
  $("btnTrain").addEventListener("click", ()=>{
    buildIndex(); save();
    $("modelStatus").textContent = "Con conocimiento";
    alert("Entrenamiento (índice) completado.");
  });
  $("topk").addEventListener("change", e=>{ state.bot.topk = Number(e.target.value)||5; save(); });
  $("threshold").addEventListener("change", e=>{ state.bot.threshold = Number(e.target.value)||0.15; save(); });

  // Archivos
  $("btnIngestFiles").addEventListener("click", async ()=>{
    const files = $("filePicker").files;
    if (!files || !files.length) return alert("Selecciona archivos primero.");
    await ingestFiles(Array.from(files));
  });
  $("filePicker").addEventListener("change", async ()=>{
    if ($("autoTrain").checked){
      const files = $("filePicker").files;
      await ingestFiles(Array.from(files));
      $("filePicker").value = "";
    }
  });

  // URLs
  $("btnAddUrl").addEventListener("click", ()=>{
    const url = $("urlInput").value.trim();
    if (!url) return;
    state.urlsQueue.push({id:nowId(), url, title:""});
    $("urlInput").value = "";
    save(); renderUrlQueue();
  });
  $("btnCrawl").addEventListener("click", async ()=>{
    if (!state.urlsQueue.length) return alert("Añade al menos una URL.");
    await ingestUrls(state.urlsQueue);
    state.urlsQueue = [];
    save(); renderUrlQueue();
  });
  $("btnClearSources").addEventListener("click", ()=>{ state.urlsQueue = []; save(); renderUrlQueue(); });

  // Buscar en corpus
  $("btnSearchCorpus").addEventListener("click", ()=>{
    const q = $("searchCorpus").value.trim();
    if (!q) return;
    const hits = searchChunks(q, state.bot.topk, state.bot.threshold);
    const list = $("corpusList");
    list.innerHTML = "";
    if (!hits.length){
      list.appendChild(el("div",{class:"muted small", text:"Sin coincidencias."}));
      return;
    }
    hits.forEach(h=>{
      const row = el("div",{class:"item"},[
        el("div",{class:"badge"}),
        el("div",{},[
          el("div",{text:h.doc.title}),
          el("div",{class:"sub", text:h.chunk.text.slice(0,220)+"…"})
        ]),
        el("div",{class:"small muted", text:`score ${h.score.toFixed(2)}`})
      ]);
      list.appendChild(row);
    });
  });

  // Reconstruir / Reset
  $("btnRebuild").addEventListener("click", ()=>{
    state.docs.forEach(d=> d.chunks=[]);
    buildIndex(); save();
    alert("Reconstruido el índice.");
  });
  $("btnReset").addEventListener("click", ()=>{
    if (!confirm("Esto borrará todo el conocimiento y configuración guardada. ¿Continuar?")) return;
    state.sources = []; state.docs = []; state.index = {vocab:new Map(), idf:new Map(), built:false};
    state.urlsQueue = []; state.chat = []; state.miniChat = []; state.qa = [];
    state.settings = { allowWeb:true, strictContext:true };
    $("ingestProgress").style.width = "0%";
    save(); renderSources(); renderCorpus(); renderUrlQueue(); renderChat(); renderMiniChat();
    $("modelStatus").textContent = "Sin entrenar";
  });

  // Toggles opcionales
  if ($("allowWeb")) $("allowWeb").addEventListener("change", e=>{ state.settings.allowWeb = !!e.target.checked; save(); });
  if ($("strictContext")) $("strictContext").addEventListener("change", e=>{ state.settings.strictContext = !!e.target.checked; save(); });

  // Exportar HTML
  if ($("btnExportHtml")) $("btnExportHtml").addEventListener("click", exportStandaloneHtml);

  // Chat tester
  if ($("send")) $("send").addEventListener("click", ()=> handleAsk("ask","tester"));
  if ($("ask")) $("ask").addEventListener("keydown", (e)=>{
    if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); handleAsk("ask","tester"); }
  });

  // Mini widget
  if ($("launcher")) $("launcher").addEventListener("click", ()=>{ $("mini").classList.add("show"); });
  if ($("closeMini")) $("closeMini").addEventListener("click", ()=>{ $("mini").classList.remove("show"); });
  if ($("miniSend")) $("miniSend").addEventListener("click", ()=> handleAsk("miniAsk","mini"));
  if ($("miniAsk")) $("miniAsk").addEventListener("keydown", (e)=>{
    if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); handleAsk("miniAsk","mini"); }
  });

  // Wizard JSONL
  bindWizardEvents();
}

/* ===================== Chat ===================== */
function pushAndRender(scope, role, text){
  const arr = (scope==="mini") ? state.miniChat : state.chat;
  arr.push({role, text});
  (scope==="mini") ? renderMiniChat() : renderChat();
  save();
}
function handleAsk(inputId, scope){
  const input = $(inputId);
  const q = input.value.trim();
  if (!q) return;
  input.value = "";

  const qLower = q.toLowerCase();
  if (/(eres|tú eres|tu eres).*(ia|inteligencia|modelo|chatgpt|gemini)/i.test(qLower)){
    pushAndRender(scope,'assistant', `Soy tu asistente virtual. ¿En qué puedo ayudarte hoy?`);
    return;
  }

  pushAndRender(scope, 'user', q);

  const qa = answerFromQA(q);
  if (qa){ pushAndRender(scope, 'assistant', qa.a + (qa.src ? `\n\nFuente: ${qa.src}` : "")); return; }

  const hits = searchChunks(q, state.bot.topk, state.bot.threshold);

  if (!hits.length){
    askServerAI(q, scope).then(ai=>{
      if (ai){ pushAndRender(scope,'assistant', ai); }
      else { pushAndRender(scope,'assistant', genericFallback(q)); }
    });
    return;
  }

  const answer = synthesizeAnswer(q, hits) || genericFallback(q);
  const styled = stylizeAnswer(answer, state.bot.system, state.bot.notes);
  pushAndRender(scope, 'assistant', styled);
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
