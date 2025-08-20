/* app.js – Studio Chatbot v2 (GENÉRICO)
   Características:
   - RAG local (TF-IDF + expansión de sinónimos)
   - Ingesta archivos/URL con limpieza + extracción
   - Wizard JSONL con autocompletar desde URL (texto claro)
   - Historial persistente
   - Fallback a IA backend (opcional) + Always-On
*/

// ==== CONFIG ====
const AI_SERVER_URL = ""; // ← déjalo vacío si no tienes backend (ej: Gemini). Pon "https://tu-backend/chat" si lo usas.
const PROXY_ORIGIN  = ""; // ← si creas un Cloudflare Worker, pon "https://tu-worker.workers.dev"

// ==== Estado ====
const state = {
  bot: { name:"", goal:"", notes:"", system:"", topk:5, threshold:0.15 },
  sources: [],
  docs: [],
  index: { vocab:new Map(), idf:new Map(), built:false },
  urlsQueue: [],
  chat: [],
  miniChat: [],
  qa: [],
  settings: { allowWeb: true, strictContext: true },
  metaDocId: null
};

// ==== Helpers ====
const $  = (id)=> document.getElementById(id);
const el = (tag, attrs={}, children=[])=>{
  const n = document.createElement(tag);
  Object.entries(attrs).forEach(([k,v])=>{
    if (k==='class') n.className = v;
    else if (k==='text') n.textContent = v;
    else n.setAttribute(k,v);
  });
  children.forEach(c=> n.appendChild(c));
  return n;
};
const nowId = ()=> Math.random().toString(36).slice(2)+Date.now().toString(36);
const uniq  = (arr)=> Array.from(new Set((arr||[]).filter(Boolean)));

const STORAGE_KEY = "studio-chatbot-v2";
function save(){
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    bot: state.bot,
    sources: state.sources,
    docs: state.docs.map(d=>({id:d.id, sourceId:d.sourceId, title:d.title, text:d.text})),
    urlsQueue: state.urlsQueue,
    qa: state.qa,
    chat: state.chat,
    miniChat: state.miniChat,
    settings: state.settings,
    metaDocId: state.metaDocId
  }));
}
function load(){
  try{
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    Object.assign(state.bot, data.bot||{});
    state.sources = data.sources||[];
    state.docs    = (data.docs||[]).map(d=>({...d, chunks:[]}));
    state.urlsQueue = data.urlsQueue||[];
    state.qa = data.qa||[];
    state.chat = data.chat||[];
    state.miniChat = data.miniChat||[];
    state.settings = data.settings || state.settings;
    state.metaDocId = data.metaDocId || null;
  }catch(e){ console.warn("No se pudo cargar estado:", e); }
}

// ==== Texto util ====
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
  return (text||"").toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g,"")
    .replace(/[^a-z0-9áéíóúñü\s]/gi,' ')
    .split(/\s+/)
    .filter(w=> w && !STOP.has(w) && w.length>1);
}
function chunkText(text, chunkSize=1200, overlap=120){
  const words = (text||"").split(/\s+/);
  const chunks = [];
  const step = Math.max(1, chunkSize - overlap);
  for (let i=0;i<words.length;i+=step){
    const part = words.slice(i, i+chunkSize).join(' ').trim();
    if (part.length>=40) chunks.push(part);
  }
  return chunks;
}
function topSentences(text, max=4, minLen=50){
  const sents = (text||"").replace(/\s+/g," ").split(/(?<=[\.\!\?])\s+/)
    .map(s=>s.trim()).filter(s=> s && s.length>=minLen && /[a-záéíóúñ]/i.test(s));
  sents.sort((a,b)=> b.length - a.length);
  return sents.slice(0, max);
}

// ==== Index (TF-IDF) ====
function buildIndex(){
  createOrUpdateMetaDoc();

  const vocab = new Map();
  const allChunks = [];
  state.docs.forEach(doc=>{
    if (!doc.chunks?.length){
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
    toks.forEach(t=> tf.set(t,(tf.get(t)||0)+1));
    const vec = new Map();
    for (const [t,f] of tf){
      const idf_t = idf.get(t)||0;
      vec.set(t, (f/toks.length)*idf_t);
    }
    ch.vector = vec;
  });

  state.index.vocab = vocab;
  state.index.idf   = idf;
  state.index.built = true;
  renderCorpus();
}
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
    const d = state.docs.find(x=> x.id===state.metaDocId);
    if (d){ d.text = text; d.chunks = []; return; }
  }
  const sid = nowId();
  state.sources.push({id:sid, type:'file', title:'(perfil del bot)', addedAt:Date.now()});
  const did = nowId();
  state.docs.push({id:did, sourceId:sid, title:'Perfil del bot', text, chunks:[]});
  state.metaDocId = did;
}

// ==== Similaridad ====
function cosineSim(a,b){
  let dot=0, na=0, nb=0;
  a.forEach((va,t)=>{ const vb=b.get(t)||0; dot+=va*vb; na+=va*va; });
  b.forEach(vb=>{ nb+=vb*vb; });
  if (na===0||nb===0) return 0;
  return dot/(Math.sqrt(na)*Math.sqrt(nb));
}
function vectorizeQuery(q){
  const tf = new Map(); const toks = tokens(q);
  toks.forEach(t=> tf.set(t,(tf.get(t)||0)+1));
  const vec = new Map();
  toks.forEach(t=>{
    const idf_t = state.index.idf.get(t)||0;
    vec.set(t, (tf.get(t)/toks.length)*idf_t);
  });
  return vec;
}

// ==== Búsqueda con sinónimos ====
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
    scored.sort((a,b)=> b.score-a.score);
    return scored.slice(0,k);
  };
  let hits = run(query, thr);
  if (!hits.length){
    const qx = expandQuery(query);
    hits = run(qx, Math.max(0.10, thr*0.7));
  }
  return hits;
}

// ==== Q&A JSONL (match previo) ====
function simQ(a,b){ return cosineSim(vectorizeQuery(a), vectorizeQuery(b)); }
function answerFromQA(query){
  if (!state.qa.length) return null;
  let best = {i:-1, score:0};
  for (let i=0;i<state.qa.length;i++){
    const s = simQ(query, state.qa[i].q);
    if (s>best.score) best = {i,score:s};
  }
  return (best.score>=0.30) ? state.qa[best.i] : null;
}

// ==== Always-On ====
function guessIntent(q){
  const s = (q||"").toLowerCase();
  if (/(precio|costo|vale|cu[aá]nto).*(plan|servicio)|\bplanes?\b/.test(s)) return 'planes';
  if (/(suscrip|registro|alta|afili|inscrib)/.test(s)) return 'alta';
  if (/(whats?app|contact|tel[eé]fono|correo|email|direcci[oó]n|ubicaci[oó]n)/.test(s)) return 'contacto';
  if (/(pol[ií]tica|garant[ií]a|devoluci[oó]n|reembolso|t[eé]rminos|condiciones)/.test(s)) return 'politicas';
  if (/(horario|atenci[oó]n|hora|agenda|cita)/.test(s)) return 'agenda';
  return 'general';
}
function generateAlwaysOnAnswer(q){
  switch (guessIntent(q)) {
    case 'planes':   return "Puedo detallar planes, precios y qué incluye cada opción. Dime qué necesitas y tu presupuesto aproximado para recomendarte mejor.";
    case 'alta':     return "Te ayudo a registrarte: dime tu nombre, correo y el servicio/plan que prefieres. Te guío paso a paso.";
    case 'contacto': return "¿Prefieres WhatsApp, teléfono o email? Te comparto los datos y, si quieres, dejo registro de tu solicitud.";
    case 'politicas':return "Te explico políticas de garantía, cambios y devoluciones con plazos y requisitos. ¿Qué caso quieres resolver?";
    case 'agenda':   return "¿Qué día y franja te sirve? Propongo horarios y confirmo la cita.";
    default:         return "Puedo ayudarte con información de servicios, precios, horarios, políticas y contacto. Cuéntame tu objetivo para darte una respuesta concreta.";
  }
}

// ==== Lectura de archivos ====
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
      let out=""; for (let i=1;i<=pdf.numPages;i++){
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const pageText = content.items.map(it=> it.str).join(" ");
        out += pageText + "\n";
      } return out;
    } else { alert("Para leer PDF, incluye pdfjs o sube texto plano."); return ""; }
  }
  alert(`Formato no soportado: .${ext}`); return "";
}

// ==== Fallbacks CORS para URL ====
async function fetchUrlText(url){
  // directo
  try{
    const res = await fetch(url, {mode:'cors'});
    if (res.ok) return await res.text();
  }catch(e){ console.warn("Fetch directo falló:", e); }

  // proxy propio
  if (PROXY_ORIGIN){
    try{
      const r = await fetch(`${PROXY_ORIGIN}/fetch?url=${encodeURIComponent(url)}`);
      if (r.ok) return await r.text();
    }catch(e){ console.warn("Proxy propio falló:", e); }
  }

  // Jina Reader (texto legible)
  try{
    const scheme = url.startsWith("https://") ? "https://" : "http://";
    const j = await fetch(`https://r.jina.ai/${scheme}${url.replace(/^https?:\/\//,'')}`);
    if (j.ok){ const t = await j.text(); if (t.trim()) return t; }
  }catch(e){ console.warn("Jina falló:", e); }

  // AllOrigins (HTML crudo)
  try{
    const ao = await fetch(`https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`);
    if (ao.ok) return await ao.text();
  }catch(e){ console.warn("AllOrigins falló:", e); }

  alert("No pude leer esa URL (CORS). Prueba otra o usa un proxy propio.");
  return "";
}

// ==== Limpieza + extracción (para el wizard) ====
function cleanForAnswer(input){
  let t = (input||"");
  t = t.replace(/https?:\/\/\S+/gi, " ");                       // URLs
  t = t.replace(/^(Title|URL Source|Published Time|Markdown Content|Image \d+):.*$/gmi, " ");
  t = t.replace(/!\[[^\]]*\]\([^)]+\)/g, " ");                  // imágenes MD
  t = t.replace(/\[[^\]]*\]\([^)]+\)/g, " ");                   // enlaces MD
  t = t.replace(/<script[\s\S]*?<\/script>/gi," ")
       .replace(/<style[\s\S]*?<\/style>/gi," ")
       .replace(/<[^>]+>/g," ")
       .replace(/&[a-z]+;/gi," ");
  t = t.replace(/este sitio utiliza cookies.*?acept(a|o)?|accept|decline/gi, " ");

  const MENU_WORDS = ["inicio","home","blog","tienda","shop","carrito","cart","mi cuenta","account","nosotros","about","servicios","productos","contacto","contact"];
  t = t.split(/\n+/).map(line=>{
    const l = line.trim().toLowerCase();
    const isMenuLike = l.length<=80 && !/[\.!\?@0-9]/.test(l) && /(\||•|·|\/|>|-|\s{2,})/.test(l) &&
      l.split(/[\s\|•·>\/-]+/).filter(Boolean).every(w=> MENU_WORDS.includes(w));
    return isMenuLike ? "" : line;
  }).filter(Boolean).join("\n");

  return t.replace(/[ \t]+/g," ").replace(/\s{2,}/g," ").trim();
}

function extractFromText(url, rawText){
  const raw = rawText || "";
  const text = cleanForAnswer(raw);

  const linesRaw   = raw.split(/\n+/).map(s=>s.trim()).filter(Boolean);
  const linesClean = text.split(/\n+/).map(s=>s.trim()).filter(Boolean);

  const emails = Array.from(raw.matchAll(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi)).map(m=>m[0]);
  const wa = Array.from(raw.matchAll(/wa\.me\/(\d{7,15})/gi)).map(m=>m[1]);
  const phoneCand = Array.from(raw.matchAll(/\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b/g)).map(m=>m[0]);
  const phones = uniq(wa.concat(phoneCand.map(p=>p.replace(/[^\d+]/g,"")))).filter(d=>{
    const len = d.replace(/\D/g,"").length; return len>=7 && len<=15;
  });

  const hours    = linesClean.filter(l=>/(horario|lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|\b\d{1,2}:\d{2}\b|\bam\b|\bpm\b)/i.test(l)).slice(0,8);
  const offers   = linesClean.filter(l=>/(precio|plan|paquete|servicio|producto|\$|\bUSD\b|\bCOP\b|\bMXN\b)/i.test(l)).slice(0,12);
  const policies = linesClean.filter(l=>/(pol[ií]tica|t[eé]rminos|condiciones|garant[ií]a|devoluci[oó]n|reembolso|privacidad)/i.test(l)).slice(0,12);

  const desc = topSentences(text, 4, 50).join(" ");

  const faqs = [];
  for (let i=0;i<linesClean.length-1;i++){
    const q = linesClean[i], a = linesClean[i+1];
    if (/\?/.test(q) && a && !/\?$/.test(a)){
      const ans = topSentences(a, 2, 30).join(" ");
      if (ans) faqs.push({ q: q.replace(/\s+/g," ").trim(), a: ans });
    }
  }

  const contactHints = [];
  const contactRaw   = linesRaw.filter(l=>/(whats?app|contacto|direcci[oó]n|ubicaci[oó]n|soporte|correo|email|tel[eé]fono|celular)/i.test(l)).slice(0,3);
  const contactClean = linesClean.filter(l=>/(whats?app|contacto|direcci[oó]n|ubicaci[oó]n|soporte|correo|email|tel[eé]fono|celular)/i.test(l)).slice(0,3);
  if (emails.length) contactHints.push(`Email(s): ${uniq(emails).join(", ")}`);
  if (phones.length) contactHints.push(`Teléfono(s): ${uniq(phones).join(", ")}`);
  contactHints.push(...contactRaw, ...contactClean);
  const contact = uniq(contactHints).slice(0,4).join(" • ");

  let name=""; try{ const host = new URL(url).hostname.replace(/^www\./,''); name = host.split('.')[0]; }catch{}
  name = name ? name.charAt(0).toUpperCase()+name.slice(1) : "";

  return { name, desc, contact, hours, offers, policies, faqs };
}

// ==== Ingesta ====
let ingestBusy=false;
function setBusy(f){
  ingestBusy=f;
  $("btnIngestFiles").disabled=f;
  $("btnCrawl").disabled=f;
  $("btnTrain").disabled=f;
  $("btnRebuild").disabled=f;
  $("btnReset").disabled=f;
}
async function ingestFiles(files){
  if (!files?.length) return;
  setBusy(true);
  const bar = $("ingestProgress"); if (bar) bar.style.width = "0%";
  let done=0;

  for (const f of files){
    const ext = (f.name.split('.').pop()||"").toLowerCase();

    if (ext==='jsonl'){
      const raw = await f.text();
      const lines = raw.split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
      const qaPairs=[];
      for (const line of lines){
        try{
          const obj = JSON.parse(line);
          if (obj && obj.q && obj.a){
            qaPairs.push({q:String(obj.q), a:String(obj.a), src: obj.src?String(obj.src):undefined, tags: Array.isArray(obj.tags)?obj.tags:undefined});
          }
        }catch{}
      }
      if (qaPairs.length){
        state.qa.push(...qaPairs);
        const txt = qaPairs.map(x=>`PREGUNTA: ${x.q}\nRESPUESTA: ${x.a}${x.src?`\nFUENTE: ${x.src}`:""}`).join("\n\n");
        const sid = nowId();
        state.sources.push({id:sid,type:'file',title:f.name,addedAt:Date.now()});
        state.docs.push({id:nowId(),sourceId:sid,title:f.name,text:txt,chunks:[]});
      }
      done++; if (bar) bar.style.width = `${Math.round(done/files.length*100)}%`;
      continue;
    }

    const text = await readFileAsText(f);
    if (!text){ done++; if (bar) bar.style.width = `${Math.round(done/files.length*100)}%`; continue; }

    const sid = nowId();
    state.sources.push({id:sid,type:'file',title:f.name,addedAt:Date.now()});
    state.docs.push({id:nowId(),sourceId:sid,title:f.name,text,chunks:[]});
    done++; if (bar) bar.style.width = `${Math.round(done/files.length*100)}%`;
  }

  buildIndex(); save(); renderSources(); setBusy(false);
  $("modelStatus").textContent = "Con conocimiento";
}

async function ingestUrls(urls){
  if (!urls?.length) return;
  setBusy(true);
  for (const u of urls){
    const raw = await fetchUrlText(u.url);
    const cleaned = cleanForAnswer(raw);
    const final = cleaned || normalizeText(raw) || raw || "";
    const sid = nowId();
    state.sources.push({id:sid,type:'url',title:u.title||u.url,href:u.url,addedAt:Date.now()});
    state.docs.push({id:nowId(),sourceId:sid,title:u.title||u.url,text:final,chunks:[]});
  }
  buildIndex(); save(); renderSources(); setBusy(false);
  $("modelStatus").textContent = "Con conocimiento";
}

// ==== Historial ====
function getHistory(scope,max=10){
  const arr = (scope==="mini")? state.miniChat : state.chat;
  return arr.slice(-max).map(m=>({role:m.role,text:m.text}));
}

// ==== Backend IA (opcional) ====
async function askServerAI(q,scope,opts={allowWeb:false}){
  if (!AI_SERVER_URL){ return null; } // si no hay backend, no lo intentes
  const lowHits = searchChunks(q,6,0.12);
  const ctx = lowHits.map(h=> h.chunk.text.slice(0,1600));
  const titles = lowHits.map(h=> h.doc.title);
  const urlSources = state.sources.filter(s=> s.type==='url' && s.href).map(s=> s.href);

  const body = {
    message:q,
    context:ctx,
    titles,
    history:getHistory(scope,10),
    profile:{name:state.bot.name,goal:state.bot.goal,notes:state.bot.notes},
    allowWeb: !!opts.allowWeb && urlSources.length>0 && !!state.settings.allowWeb,
    webUrls: (!!opts.allowWeb ? urlSources.slice(0,3): []),
    strictContext: !!state.settings.strictContext
  };
  try{
    const r = await fetch(AI_SERVER_URL,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const json = await r.json();
    return (json.answer||"").trim();
  }catch(e){ console.warn("AI server error:", e); return null; }
}

// ==== Síntesis RAG ====
function synthesizeAnswer(query,hits){
  const sentences=[]; const seen=new Set();
  hits.forEach(h=>{
    h.chunk.text.split(/(?<=[\.\!\?])\s+/).forEach(s=>{
      const t=s.trim(); if(!t||t.length<30) return;
      const key=t.toLowerCase(); if(seen.has(key)) return; seen.add(key);
      sentences.push({text:t,score:h.score});
    });
  });
  sentences.sort((a,b)=> b.score-a.score);
  const picked = sentences.slice(0,6).map(s=>s.text);
  if(!picked.length) return "";
  const bullets = picked.map(s=>"• "+s);
  const srcTitles = Array.from(new Set(hits.map(h=>h.doc.title))).slice(0,3);
  return [bullets.join("\n"), srcTitles.length?`Fuentes: ${srcTitles.join(" • ")}`:""].filter(Boolean).join("\n\n");
}
function stylizeAnswer(text,system,notes){
  let t=text;
  if (notes && /breve|conciso/i.test(notes) && t.length>600) t = t.slice(0,600)+"…";
  if (system) t = t.replaceAll(system,"");
  return t;
}

// ==== Render ====
function renderBasics(){
  $("botName").value = state.bot.name||"";
  $("botGoal").value = state.bot.goal||"";
  $("botNotes").value = state.bot.notes||"";
  $("systemPrompt").value = state.bot.system||"";
  $("topk").value = state.bot.topk;
  $("threshold").value = state.bot.threshold;
  $("allowWeb").checked = !!state.settings.allowWeb;
  $("strictContext").checked = !!state.settings.strictContext;
  $("botNameDisplay").textContent = state.bot.name || "(sin nombre)";
  $("botGoalDisplay").textContent = state.bot.goal || "";
  $("miniTitle").textContent = state.bot.name || "Asistente";
  $("modelStatus").textContent = state.docs.length ? "Con conocimiento" : "Sin entrenar";
  $("embedSnippet").textContent =
`<!-- Bot Studio v2 -->
<link rel="stylesheet" href="(usa tu mismo CSS)">
<div class="launcher" id="launcher">💬</div>
<div class="mini" id="mini"> ... </div>
<script src="app.js"></script>`;
}
function renderSources(){
  const list=$("sourcesList"); list.innerHTML="";
  if (!state.sources.length){ list.appendChild(el("div",{class:"muted small",text:"Aún no has cargado fuentes."})); return; }
  state.sources.slice().sort((a,b)=>b.addedAt-a.addedAt).forEach(s=>{
    const row = el("div",{class:"item"},[
      el("div",{class:"badge"}),
      el("div",{},[ el("div",{text:s.title}), el("div",{class:"small muted",text:s.type==='url'?(s.href||'URL'):'Archivo'}) ]),
      s.href? el("a",{href:s.href,target:"_blank",class:"small muted",text:"Ver"}) : el("span",{class:"small muted",text:""})
    ]);
    list.appendChild(row);
  });
}
function renderCorpus(){
  const list=$("corpusList"); list.innerHTML="";
  if (!state.docs.length){ list.appendChild(el("div",{class:"muted small",text:"Sin documentos. Sube archivos o añade URLs."})); return; }
  state.docs.forEach(d=>{
    const lines = d.text.split(/\n/).slice(0,3).join(" ").slice(0,140);
    const row = el("div",{class:"item"},[
      el("div",{class:"badge"}),
      el("div",{},[ el("div",{text:d.title}), el("div",{class:"sub",text: lines+(d.text.length>140?'…':'')}) ]),
      el("span",{class:"small muted",text:`${(d.chunks?.length)||0} chunks`})
    ]);
    list.appendChild(row);
  });
}
function renderUrlQueue(){
  const list=$("urlList"); list.innerHTML="";
  if (!state.urlsQueue.length){ list.appendChild(el("div",{class:"muted small",text:"No hay URLs en cola."})); return; }
  state.urlsQueue.forEach(u=>{
    const row = el("div",{class:"item"},[
      el("div",{class:"badge"}),
      el("div",{},[ el("div",{text:u.title||u.url}), el("div",{class:"sub",text:u.url}) ]),
      el("button",{class:"ghost small",text:"Quitar"})
    ]);
    row.querySelector("button").addEventListener("click",()=>{
      state.urlsQueue = state.urlsQueue.filter(x=> x.id!==u.id); save(); renderUrlQueue();
    });
    list.appendChild(row);
  });
}
function renderChat(){
  const log=$("chatlog"); log.innerHTML="";
  state.chat.forEach(m=>{
    const b=el("div",{class:`bubble ${m.role==='user'?'user':'bot'}`}); b.textContent=m.text; log.appendChild(b);
  });
  log.scrollTop = log.scrollHeight;
}
function renderMiniChat(){
  const log=$("miniLog"); log.innerHTML="";
  state.miniChat.forEach(m=>{
    const b=el("div",{class:`bubble ${m.role==='user'?'user':'bot'}`}); b.textContent=m.text; log.appendChild(b);
  });
  log.scrollTop = log.scrollHeight;
}

// ==== Chat ====
function pushAndRender(scope,role,text){
  const arr = (scope==="mini")? state.miniChat : state.chat;
  arr.push({role,text});
  (scope==="mini")? renderMiniChat() : renderChat();
  save();
}
function handleAsk(inputId, logId, scope){
  const input = $(inputId);
  const q = (input.value||"").trim();
  if (!q) return;
  input.value="";

  const qLower = q.toLowerCase();
  if (/(eres|tú eres|tu eres).*(gemini|ia|inteligencia|chatgpt|modelo)/i.test(qLower)){
    pushAndRender(scope,'assistant',`Soy el asistente virtual de ${state.bot.name || "este sitio"}. ¿Te ayudo con algo específico?`);
    return;
  }

  pushAndRender(scope,'user',q);

  const qa = answerFromQA(q);
  if (qa){ pushAndRender(scope,'assistant', qa.a + (qa.src?`\n\nFuente: ${qa.src}`:"")); return; }

  const hits = searchChunks(q, state.bot.topk, state.bot.threshold);

  if (!hits.length){
    askServerAI(q,scope,{allowWeb:true}).then(ai=>{
      if (ai){ pushAndRender(scope,'assistant',ai); }
      else { pushAndRender(scope,'assistant', generateAlwaysOnAnswer(q)); }
    });
    return;
  }

  const answer = synthesizeAnswer(q,hits) || generateAlwaysOnAnswer(q);
  pushAndRender(scope,'assistant', stylizeAnswer(answer, state.bot.system, state.bot.notes));
}

// ==== Wizard JSONL ====
function buildJSONLPairsFromWizard(){
  const name = ($("w_name")?.value||"").trim();
  const tone = ($("w_tone")?.value||"Cercano y profesional.").trim();
  const desc = ($("w_desc")?.value||"").trim();
  const contact = ($("w_contact")?.value||"").trim();
  const hours = ($("w_hours")?.value||"").trim().split(/\n+/).filter(Boolean);
  const offers = ($("w_offers")?.value||"").trim().split(/\n+/).filter(Boolean);
  const policies = ($("w_policies")?.value||"").trim().split(/\n+/).filter(Boolean);
  const faqsLines = ($("w_faqs")?.value||"").trim().split(/\n+/).filter(Boolean);

  const pairs=[];
  if (desc){
    pairs.push({ q:`¿Qué es ${name||'la empresa'} y qué hace?`, a:`${desc}\n\nTono del asistente: ${tone}`, src:"perfil" });
  }
  if (contact){
    pairs.push({ q:"¿Cómo puedo contactarlos?", a: `${contact}\n\nTono del asistente: ${tone}`, src:"contacto" });
  }
  if (hours.length){
    pairs.push({ q:"¿Cuáles son los horarios de atención?", a: hours.join(" • "), src:"operación" });
  }
  if (offers.length){
    pairs.push({ q:"¿Qué productos/servicios ofrecen y precios?", a: offers.join(" • "), src:"oferta" });
  }
  if (policies.length){
    pairs.push({ q:"¿Cuáles son sus políticas (garantía, cambios, devoluciones)?", a: policies.join(" • "), src:"políticas" });
  }
  for (const line of faqsLines){
    const [q,a] = line.split("|").map(s=>(s||"").trim());
    if (q && a) pairs.push({ q, a, src:"faq" });
  }
  if (!pairs.length){
    pairs.push({ q:"¿Qué ofrecen?", a:"Ofrecemos servicios y soluciones adaptadas a tu necesidad. Cuéntame tu caso y preparo una propuesta clara con pasos y precios.", src:"fallback" });
  }
  return pairs;
}
function previewJSONL(){
  const pairs = buildJSONLPairsFromWizard();
  $("jsonlPreview").value = pairs.map(p=> JSON.stringify(p)).join("\n");
}
function downloadJSONL(){
  const txt = $("jsonlPreview").value.trim();
  if (!txt) return alert("Previsualiza antes de descargar.");
  const blob = new Blob([txt], {type:"application/jsonl"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "dataset_chatbot.jsonl"; a.click();
  URL.revokeObjectURL(a.href);
}
function addJSONLToProject(){
  const txt = $("jsonlPreview").value.trim();
  if (!txt) return alert("Previsualiza antes de agregar.");
  const lines = txt.split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
  const qaPairs=[];
  for (const line of lines){
    try{
      const obj = JSON.parse(line);
      if (obj && obj.q && obj.a){
        qaPairs.push({q:String(obj.q), a:String(obj.a), src: obj.src?String(obj.src):undefined, tags: Array.isArray(obj.tags)?obj.tags:undefined});
      }
    }catch{}
  }
  if (!qaPairs.length) return alert("No se encontraron pares válidos.");
  state.qa.push(...qaPairs);
  const txtDoc = qaPairs.map(x=>`PREGUNTA: ${x.q}\nRESPUESTA: ${x.a}${x.src?`\nFUENTE: ${x.src}`:""}`).join("\n\n");
  const sid = nowId();
  state.sources.push({id:sid,type:'file',title:'dataset_chatbot.jsonl',addedAt:Date.now()});
  state.docs.push({id:nowId(),sourceId:sid,title:'dataset_chatbot.jsonl',text:txtDoc,chunks:[]});
  buildIndex(); save(); renderSources(); renderCorpus(); alert("Dataset agregado e indexado.");
}

// Autocompletar desde URL
async function autoFillFromUrl(){
  const elUrl = $("wizardUrlInput"); if (!elUrl) return;
  const url = elUrl.value.trim(); if (!url) return alert("Pega una URL primero.");
  const txt = await fetchUrlText(url); if (!txt) return;

  const b = extractFromText(url, txt);
  if (b.name && $("w_name")) $("w_name").value = b.name;
  if ($("w_tone")) $("w_tone").value = $("w_tone").value || "Cercano y profesional.";
  if (b.desc && $("w_desc")) $("w_desc").value = b.desc;
  if (b.contact && $("w_contact")) $("w_contact").value = b.contact;
  if (b.hours?.length && $("w_hours")) $("w_hours").value = b.hours.join("\n");
  if (b.offers?.length && $("w_offers")) $("w_offers").value = b.offers.join("\n");
  if (b.policies?.length && $("w_policies")) $("w_policies").value = b.policies.join("\n");
  if (b.faqs?.length && $("w_faqs")) $("w_faqs").value = b.faqs.map(f=>`${f.q} | ${f.a}`).join("\n");

  // también precarga el panel del bot si está vacío
  if (b.name && !$("botName").value){ $("botName").value=b.name; state.bot.name=b.name; }
  if (b.desc && !$("botGoal").value){ $("botGoal").value=b.desc; state.bot.goal=b.desc; }
  if (!$("botNotes").value){
    $("botNotes").value = "Tono: " + (($("w_tone")?.value)||"Cercano y profesional.");
    state.bot.notes = $("botNotes").value;
  }
  save(); renderBasics();
  alert("Campos autocompletados. Revisa/edita y genera tu JSONL.");
}

// Export HTML
function exportStaticHTML(){
  const html = document.documentElement.outerHTML;
  const blob = new Blob([html], {type:"text/html;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "chatbot_export.html"; a.click();
  URL.revokeObjectURL(a.href);
}

// ==== Eventos ====
function bindEvents(){
  $("botName").addEventListener("input", e=>{
    state.bot.name = e.target.value;
    $("botNameDisplay").textContent = state.bot.name || "(sin nombre)";
    $("miniTitle").textContent = state.bot.name || "Asistente";
    save();
  });
  $("botGoal").addEventListener("input", e=>{ state.bot.goal = e.target.value; $("botGoalDisplay").textContent = state.bot.goal || ""; save(); });
  $("botNotes").addEventListener("input", e=>{ state.bot.notes = e.target.value; save(); });
  $("systemPrompt").addEventListener("input", e=>{ state.bot.system = e.target.value; save(); });

  $("allowWeb").addEventListener("change", e=>{ state.settings.allowWeb = e.target.checked; save(); });
  $("strictContext").addEventListener("change", e=>{ state.settings.strictContext = e.target.checked; save(); });

  $("topk").addEventListener("change", e=>{ state.bot.topk = Number(e.target.value)||5; save(); });
  $("threshold").addEventListener("change", e=>{ state.bot.threshold = Number(e.target.value)||0.15; save(); });

  $("btnIngestFiles").addEventListener("click", async ()=>{
    const files = $("filePicker").files;
    if (!files || !files.length) return alert("Selecciona archivos primero.");
    await ingestFiles(Array.from(files));
  });
  $("filePicker").addEventListener("change", async ()=>{
    if ($("autoTrain").checked){
      const files = $("filePicker").files;
      await ingestFiles(Array.from(files));
      $("filePicker").value="";
    }
  });

  $("btnAddUrl").addEventListener("click", ()=>{
    const url = $("urlInput").value.trim();
    if (!url) return;
    state.urlsQueue.push({id:nowId(), url, title:""}); $("urlInput").value="";
    save(); renderUrlQueue();
  });
  $("btnCrawl").addEventListener("click", async ()=>{
    if (!state.urlsQueue.length) return alert("Añade al menos una URL.");
    await ingestUrls(state.urlsQueue); state.urlsQueue=[]; save(); renderUrlQueue();
  });
  $("btnClearSources").addEventListener("click", ()=>{ state.urlsQueue=[]; save(); renderUrlQueue(); });

  $("btnSearchCorpus").addEventListener("click", ()=>{
    const q = $("searchCorpus").value.trim(); if (!q) return;
    const hits = searchChunks(q, state.bot.topk, state.bot.threshold);
    const list = $("corpusList"); list.innerHTML="";
    if (!hits.length){ list.appendChild(el("div",{class:"muted small",text:"Sin coincidencias."})); return; }
    hits.forEach(h=>{
      const row = el("div",{class:"item"},[
        el("div",{class:"badge"}),
        el("div",{},[ el("div",{text:h.doc.title}), el("div",{class:"sub",text:h.chunk.text.slice(0,220)+"…"}) ]),
        el("div",{class:"small muted",text:`score ${h.score.toFixed(2)}`})
      ]); list.appendChild(row);
    });
  });

  $("btnTrain").addEventListener("click", ()=>{ buildIndex(); save(); $("modelStatus").textContent="Con conocimiento"; alert("Entrenamiento (índice) completado."); });
  $("btnRebuild").addEventListener("click", ()=>{ state.docs.forEach(d=> d.chunks=[]); buildIndex(); save(); alert("Reconstruido el índice."); });
  $("btnReset").addEventListener("click", ()=>{
    if (!confirm("Esto borrará todo. ¿Continuar?")) return;
    state.sources=[]; state.docs=[]; state.index={vocab:new Map(),idf:new Map(),built:false};
    state.urlsQueue=[]; state.chat=[]; state.miniChat=[]; state.qa=[];
    state.settings={allowWeb:true,strictContext:true}; state.metaDocId=null;
    $("ingestProgress").style.width="0%";
    save(); renderSources(); renderCorpus(); renderUrlQueue(); renderChat(); renderMiniChat();
    $("modelStatus").textContent="Sin entrenar";
  });

  $("btnExportHtml").addEventListener("click", exportStaticHTML);

  $("send").addEventListener("click", ()=> handleAsk("ask","chatlog","tester"));
  $("ask").addEventListener("keydown", e=>{
    if (e.key==="Enter" && !e.shiftKey){ e.preventDefault(); handleAsk("ask","chatlog","tester"); }
  });

  $("launcher").addEventListener("click", ()=>{ $("mini").classList.add("show"); });
  $("closeMini").addEventListener("click", ()=>{ $("mini").classList.remove("show"); });
  $("miniSend").addEventListener("click", ()=> handleAsk("miniAsk","miniLog","mini"));
  $("miniAsk").addEventListener("keydown", e=>{
    if (e.key==="Enter" && !e.shiftKey){ e.preventDefault(); handleAsk("miniAsk","miniLog","mini"); }
  });

  // Wizard
  const btnAuto = $("btnAutoFillFromUrl"); if (btnAuto) btnAuto.addEventListener("click", autoFillFromUrl);
  $("btnPreviewJSONL").addEventListener("click", previewJSONL);
  $("btnDownloadJSONL").addEventListener("click", downloadJSONL);
  $("btnAddJSONLToProject").addEventListener("click", addJSONLToProject);
}

// ==== Init ====
(function init(){
  load();
  renderBasics(); renderSources(); renderCorpus(); renderUrlQueue(); renderChat(); renderMiniChat();
  bindEvents();
  if (state.docs.length && !state.index.built) buildIndex();
})();
