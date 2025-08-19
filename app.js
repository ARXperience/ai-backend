/* app.js – Studio Chatbot v2 (ANÁLISIS + SEMÁNTICA + SIN SILENCIOS)
   - Análisis de contenido: extrae contacto, horarios, ofertas/precios, políticas, productos/servicios, FAQs y frases clave
   - Expansión semántica: sinónimos, variantes y corrección difusa (Levenshtein)
   - Búsqueda híbrida: TF-IDF + similitud difusa + degradación de umbral
   - Composición de respuesta basada en KB + evidencias (sin fallback vacío)
   - Wizard JSONL + Autocompletar desde URL + Exportar HTML estático
*/

/* ===== Config backend IA (opcional). Deja "" si no tienes ===== */
const AI_SERVER_URL = ""; // p.ej. "https://api.tu-dominio.com/chat"

/* ===================== Estado ===================== */
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
  // Nuevo: KB estructurado tras el análisis
  kb: {
    contact: [], hours: [], prices: [], policies: [],
    products: [], services: [], locations: [],
    faqs: [], // {q,a}
    sentences: [], // {text,weight,doc}
    keyphrases: new Map() // term -> weight
  },
  // cache de vocab y sinónimos para expansión
  vocabSet: new Set()
};

let ingestBusy = false;

/* ===================== Utils DOM ===================== */
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
function save(){
  const toSave = {
    bot: state.bot, sources: state.sources,
    docs: state.docs.map(d=>({ id:d.id, sourceId:d.sourceId, title:d.title, text:d.text, meta: !!d.meta })),
    urlsQueue: state.urlsQueue, qa: state.qa, chat: state.chat, miniChat: state.miniChat,
    settings: state.settings, kb: state.kb // persistimos análisis
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
  }catch(e){ console.warn("load error", e); }
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
  // stemming MUY ligero para español (solo las variantes más frecuentes)
  let s = w;
  s = s.replace(/(mente|ciones|ciones|cion|ciones)$/,'');
  s = s.replace(/(idades|idad|osos?|osas?)$/,'');
  s = s.replace(/(ando|iendo|ados?|idas?)$/,'');
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

/* ===================== Índice TF-IDF ===================== */
function upsertMetaDoc(){
  state.docs = state.docs.filter(d=> !d.meta);
  state.sources = state.sources.filter(s=> s.type!=='meta');

  const pieces = [];
  if (state.bot.goal) pieces.push(`OBJETIVO:\n${state.bot.goal}`);
  if (state.bot.notes) pieces.push(`NOTAS:\n${state.bot.notes}`);
  if (state.bot.system) pieces.push(`SISTEMA:\n${state.bot.system}`);
  if (!pieces.length) return;

  const text = pieces.join("\n\n");
  const sid = nowId();
  state.sources.push({ id:sid, type:'meta', title:'Perfil del bot', addedAt:Date.now() });
  state.docs.push({ id:nowId(), sourceId:sid, title:'Perfil del bot', text, meta:true, chunks:[] });
}
function buildIndex(){
  upsertMetaDoc();
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
        const st = stemEs(tok);
        if (!seen.has(st)){
          vocab.set(st, (vocab.get(st)||0)+1);
          seen.add(st);
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
    const toks = tokens(ch.text).map(stemEs);
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

  // Vocab set para expansión
  state.vocabSet = new Set(Array.from(vocab.keys()));
  renderCorpus();

  // Ejecuta análisis semántico tras indexar
  runSemanticAnalysis();
  save();
}

/* ===================== Expansión semántica ===================== */
const SYN = {
  precio:["costo","tarifa","valor","vale","cuanto","cotizacion","presupuesto"],
  horario:["hora","apertura","atencion","cierre","dias","sabado","domingo"],
  contacto:["whatsapp","telefono","llamar","celular","correo","email","direccion","ubicacion","soporte"],
  envio:["entrega","shipping","despacho","reparto","mensajeria","domicilio","tracking","seguimiento"],
  garantia:["garantia","cambios","devolucion","reembolso","tyc","terminos","condiciones","politica","privacidad"],
  producto:["servicio","oferta","plan","paquete","catalogo","portafolio"],
  ubicacion:["sede","oficina","tienda","local"],
  pago:["pagar","medio","metodo","credito","debito","transferencia","efecty","paypal"],
  pedido:["orden","compra","carrito","checkout"],
  soporte:["ayuda","asistencia","ticket","incidencia","reclamo","pqrs"],
  precio_plural:["precios","costos","tarifas","valores"],
  promo:["descuento","promocion","oferta","cupon","beneficio"],
  tiempo:["plazo","demora","tarda","entrega","estimado"],
  calidad:["original","certificado","garantizado"],
  ubicuidad:["ciudad","pais","zona","cobertura"]
};
function getSynonyms(t){
  const out = new Set([t]);
  Object.keys(SYN).forEach(k=>{
    if (k===t || SYN[k].includes(t)) { SYN[k].forEach(x=>out.add(x)); }
  });
  // plurales/singulares simples
  if (t.endsWith('s')) out.add(t.replace(/s$/,''));
  else out.add(t+'s');
  // variantes acento
  if (/[aeiou]/.test(t)) out.add(stripAcc(t));
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
    // corrección difusa: acercar a términos del vocab
    let best = null, bestD=3; // distancia máx 2
    for (const v of state.vocabSet){
      const d = levenshtein(t, v);
      if (d < bestD){ bestD = d; best = v; if (d===0) break; }
    }
    if (best && bestD<=2) expanded.add(best);
  });
  return Array.from(expanded);
}
function vectorizeQuery(q){
  const ex = expandQueryTerms(q);
  const tf = new Map();
  ex.forEach(t=> tf.set(t, (tf.get(t)||0)+1));
  const vec = new Map();
  const n = Math.max(1, ex.length);
  ex.forEach(t=>{
    const idf_t = state.index.idf.get(t) || 0;
    vec.set(t, (tf.get(t)/n) * (idf_t || 0.5)); // peso mínimo si no está en idf
  });
  return vec;
}

/* ===================== Búsqueda híbrida ===================== */
function cosineSim(a,b){
  let dot=0, na=0, nb=0;
  a.forEach((va, t)=>{ const vb=b.get(t)||0; dot += va*vb; na += va*va; });
  b.forEach(vb=>{ nb += vb*vb; });
  if (na===0 || nb===0) return 0;
  return dot / (Math.sqrt(na)*Math.sqrt(nb));
}
function tokenSet(s){ return new Set(tokens(s).map(stemEs)); }
function jaccard(aSet, bSet){
  let inter=0; bSet.forEach(x=>{ if (aSet.has(x)) inter++; });
  const union = aSet.size + bSet.size - inter;
  return union? inter/union : 0;
}
function hybridScore(q, chunk){
  const qv = vectorizeQuery(q);
  const cos = cosineSim(qv, chunk.vector);
  // componente difusa: jaccard entre query expandida y chunk tokens
  const qset = new Set(expandQueryTerms(q));
  const cset = tokenSet(chunk.text);
  const jac = jaccard(qset, cset);
  return (cos*0.7) + (jac*0.3);
}
function searchChunks(query, k=5, thr=0.15, degrade=false){
  if (!state.index.built) buildIndex();
  const scored = [];
  state.docs.forEach(doc=>{
    doc.chunks.forEach(ch=>{
      const s = hybridScore(query, ch);
      if (s>=thr) scored.push({chunk:ch, score:s, doc});
    });
  });
  scored.sort((a,b)=> b.score - a.score);
  if (scored.length) return scored.slice(0,k);
  if (!degrade){
    // degradación: baja umbral y sube k
    return searchChunks(query, Math.max(k,8), Math.max(0.08, thr*0.66), true);
  }
  return []; // no hay nada realmente
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
    } else {
      alert("Para leer PDF local, incluye pdfjs (pdfjsLib) o convierte a .txt/.md.");
      return "";
    }
  }
  alert(`Formato no soportado: .${ext}. Convierte a .txt/.md/.pdf.`);
  return "";
}
// Fetch con anti-CORS escalonado
async function fetchUrlText(url){
  try{
    const r = await fetch(url, { mode:'cors' });
    const ct = (r.headers.get('content-type')||"").toLowerCase();
    const raw = await r.text();
    if (ct.includes("html")) return normalizeText(raw);
    return raw;
  }catch{}
  try{
    const cleanURL = url.replace(/^https?:\/\//,'');
    const r = await fetch(`https://r.jina.ai/http://${cleanURL}`);
    const raw = await r.text();
    if (raw && raw.length>50) return normalizeText(raw);
  }catch{}
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
  const bar = $("ingestProgress"); bar.style.width="0%";
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

/* ===================== Análisis semántico (KB) ===================== */
function pushFact(arr, v){ if (v && typeof v==='string'){ const s=v.trim(); if (s && !arr.includes(s)) arr.push(s); } }
function runSemanticAnalysis(){
  // reset
  state.kb = { contact:[], hours:[], prices:[], policies:[], products:[], services:[], locations:[], faqs:[], sentences:[], keyphrases:new Map() };

  const addKey = (term, w=1)=>{
    const k = stemEs(stripAcc(term.toLowerCase()));
    state.kb.keyphrases.set(k, (state.kb.keyphrases.get(k)||0)+w);
  };

  state.docs.forEach(doc=>{
    const text = doc.text || "";
    const lines = text.split(/\n+/).map(s=>s.trim()).filter(Boolean);

    // Contacto
    const emails = Array.from((text||"").matchAll(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi)).map(m=>m[0]);
    const phones = Array.from((text||"").matchAll(/(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,3}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}/g))
      .map(m=>m[0]).filter(x=>x.replace(/\D/g,'').length>=7);
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
    lines.filter(l=>/(pol[ií]tica|t[eé]rminos|condiciones|garant[ií]a|devoluci[oó]n|cambios|privacidad|reembolso)/i.test(l))
      .slice(0,20).forEach(s=>pushFact(state.kb.policies, s));

    // Productos/Servicios/Ubicaciones
    lines.filter(l=>/(producto|servicio|portafolio|cat[aá]logo|oferta)/i.test(l)).slice(0,20).forEach(s=>pushFact(state.kb.products, s));
    lines.filter(l=>/(servicio|asesor[ií]a|soporte|mantenimiento|implementaci[oó]n)/i.test(l)).slice(0,20).forEach(s=>pushFact(state.kb.services, s));
    lines.filter(l=>/(sede|oficina|tienda|local|ciudad|direcci[oó]n|ubicaci[oó]n)/i.test(l)).slice(0,20).forEach(s=>pushFact(state.kb.locations, s));

    // FAQs simples: línea con ? + respuesta siguiente
    for (let i=0;i<lines.length-1;i++){
      const q = lines[i]; const a = lines[i+1];
      if (/\?/.test(q) && a && !/\?$/.test(a)){
        state.kb.faqs.push({ q:q.replace(/\s+/g," ").trim(), a:a.replace(/\s+/g," ").trim() });
      }
    }

    // Frases clave (para síntesis)
    const sents = text.split(/(?<=[\.\!\?])\s+/).map(s=>s.trim()).filter(x=>x.length>40);
    sents.slice(0,120).forEach(s=>{
      const weight = Math.min(1.0, (tokens(s).length/25));
      state.kb.sentences.push({ text:s, weight, doc:doc.title });
      tokens(s).forEach(t=> addKey(t, 0.1));
    });

    // Keyphrases por puntuación sencilla (tf)
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

/* ===================== Cliente backend IA (opcional) ===================== */
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
    draft, // pasamos borrador para que lo pula/parafrasee
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

/* ===================== Composición de respuesta ===================== */
function detectTheme(q){
  const s = stripAcc(q.toLowerCase());
  if (/(precio|costo|tarifa|cuanto|cotiza|presupuesto|plan|paquete)/.test(s)) return "prices";
  if (/(horario|atencion|abre|cierra|dias|agenda|cita)/.test(s)) return "hours";
  if (/(contacto|whats|telefono|correo|email|direccion|ubicacion|sede)/.test(s)) return "contact";
  if (/(politica|garantia|devolucion|reembolso|termino|condicion|privacidad|tyc)/.test(s)) return "policies";
  if (/(producto|servicio|portafolio|catalogo|oferta)/.test(s)) return "products";
  if (/(soporte|ayuda|ticket|incidencia|pqrs|reclamo)/.test(s)) return "support";
  if (/(envio|entrega|despacho|domicilio|tracking|seguimiento)/.test(s)) return "shipping";
  return "general";
}
function composeFromKB(theme){
  const KB = state.kb;
  const blocks = [];
  if (theme==="contact" && KB.contact.length){
    blocks.push("Contacto:\n- " + KB.contact.slice(0,3).join("\n- "));
  }
  if (theme==="hours" && KB.hours.length){
    blocks.push("Horarios/Cobertura:\n- " + KB.hours.slice(0,5).join("\n- "));
  }
  if ((theme==="prices"||theme==="products") && (KB.prices.length || KB.products.length)){
    if (KB.products.length) blocks.push("Productos/Servicios:\n- " + KB.products.slice(0,6).join("\n- "));
    if (KB.prices.length) blocks.push("Precios/Planes:\n- " + KB.prices.slice(0,6).join("\n- "));
  }
  if (theme==="policies" && KB.policies.length){
    blocks.push("Políticas / TyC:\n- " + KB.policies.slice(0,6).join("\n- "));
  }
  if (KB.faqs.length && (theme==="general"||theme==="support")){
    blocks.push("FAQ relacionada:\n- " + KB.faqs.slice(0,3).map(f=>`${f.q} → ${f.a}`).join("\n- "));
  }
  return blocks.join("\n\n");
}
function composeFromHits(q, hits){
  if (!hits.length) return "";
  // seleccionamos frases de los mejores chunks (ya filtradas por análisis)
  const sentences = [];
  const seen = new Set();
  for (const h of hits){
    h.chunk.text.split(/(?<=[\.\!\?])\s+/).forEach(s=>{
      const t = s.trim(); if (t.length<40) return;
      const key = t.toLowerCase(); if (seen.has(key)) return; seen.add(key);
      sentences.push({t, sc:h.score, doc:h.doc.title});
    });
  }
  sentences.sort((a,b)=> b.sc - a.sc);
  const picked = sentences.slice(0,6);
  const bullets = picked.map(x=>`• ${x.t}`).join("\n");
  const src = Array.from(new Set(picked.map(x=>x.doc))).slice(0,3);
  return `${bullets}${src.length?`\n\nFuentes: ${src.join(" • ")}`:""}`;
}
function composeAnswer(q, hits){
  const theme = detectTheme(q);
  const kbBlock = composeFromKB(theme);
  const hitsBlock = composeFromHits(q, hits);
  // Si no hay nada en KB para ese tema, igual usamos hits; y viceversa
  if (kbBlock && hitsBlock) return `${kbBlock}\n\n${hitsBlock}`;
  if (kbBlock) return kbBlock;
  if (hitsBlock) return hitsBlock;

  // Último recurso: resumen guiado con keyphrases
  const kp = Array.from(state.kb.keyphrases.entries()).sort((a,b)=> b[1]-a[1]).slice(0,10).map(x=>x[0]);
  const support = state.kb.sentences.slice(0,5).map(s=>`• ${s.text}`).join("\n");
  return `Relacionado con tu consulta, estos son puntos clave: ${kp.join(", ")}.\n\n${support}`;
}

/* ===================== Exportar HTML estático (igual que antes) ===================== */
function exportStandaloneHtml(){
  const payload = {
    meta: { exportedAt: new Date().toISOString(), app: "Studio Chatbot v2" },
    bot: state.bot, qa: state.qa,
    docs: state.docs.map(d => ({ title: d.title, text: d.text })), // incluye meta
  };
  const html = `<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>${(state.bot.name||"Asistente")} — Chat</title><style>:root{--bg:#0f1221;--text:#e7eaff;--brand:#6c8cff;--accent:#22d3ee}*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:radial-gradient(1000px 500px at 10% -10%,rgba(108,140,255,.15),transparent),radial-gradient(800px 400px at 90% -10%,rgba(34,211,238,.08),transparent),var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Helvetica Neue,Noto Sans,Arial}.wrap{max-width:900px;margin:0 auto;padding:20px}.card{border:1px solid rgba(255,255,255,.08);border-radius:16px;background:rgba(255,255,255,.04);padding:16px}.header{display:flex;gap:10px;align-items:center;margin-bottom:12px}.logo{width:28px;height:28px;border-radius:8px;background:conic-gradient(from 200deg at 60% 40%,var(--brand),var(--accent))}.title{font-weight:700}.chatlog{min-height:60vh;display:flex;flex-direction:column;gap:8px;overflow:auto;padding:6px}.bubble{max-width:80%;padding:10px 12px;border-radius:14px}.user{align-self:flex-end;background:rgba(108,140,255,.18);border:1px solid rgba(108,140,255,.45)}.bot{align-self:flex-start;background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.45)}.composer{display:flex;gap:10px;margin-top:10px}.composer input{flex:1;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:rgba(5,8,18,.6);color:var(--text)}.composer button{padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:linear-gradient(180deg,rgba(108,140,255,.25),rgba(108,140,255,.06));color:var(--text)}</style></head><body><div class="wrap"><div class="card"><div class="header"><div class="logo"></div><div><div class="title">${(state.bot.name||"Asistente")}</div><div style="opacity:.7">${state.bot.goal||""}</div></div></div><div id="log" class="chatlog"></div><div class="composer"><input id="ask" placeholder="Escribe..."/><button id="send">Enviar</button></div><div style="opacity:.7;margin-top:6px">Modo: RAG local (offline) • Responde siempre</div></div></div><script>window.BOOT=${JSON.stringify(payload)};</script><script>(function(){const STOP=new Set("a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos el ella ellas ellos en entre era erais éramos eran es esa esas ese esos esta estaba estabais estábamos estaban estar este esto estos fue fui fuimos ha han hasta hay la las le les lo los mas más me mientras muy nada ni nos o os otra otros para pero poco por porque que quien se ser si sí sin sobre soy su sus te tiene tengo tuvo tuve u un una unas unos y ya".split(/\\s+/));const strip=s=>s.normalize('NFD').replace(/[\\u0300-\\u036f]/g,"");const stem=w=>strip(w).toLowerCase().replace(/(mente|ciones|cion|idades|idad|osos?|osas?|ando|iendo|ados?|idas?|es|s)$/,'');const tokens=t=>strip(t.toLowerCase()).replace(/[^a-z0-9áéíóúñü\\s]/gi,' ').split(/\\s+/).filter(w=>w&&!STOP.has(w)&&w.length>1);const chunk=(txt,sz=1200,ov=120)=>{const w=txt.split(/\\s+/);const out=[];for(let i=0;i<w.length;i+=Math.max(1,Math.floor(sz-ov))){const part=w.slice(i,i+sz).join(' ').trim();if(part.length>40) out.push(part);}return out};function build(docs){const vocab=new Map();const chs=[];docs.forEach(d=>{chunk(d.text).forEach((t,i)=>{const c={id:d.title+"#"+i,text:t};chs.push(c);const seen=new Set();tokens(t).map(stem).forEach(tok=>{if(!seen.has(tok)){vocab.set(tok,(vocab.get(tok)||0)+1);seen.add(tok);}});});});const N=chs.length||1;const idf=new Map();for(const [term,df] of vocab) idf.set(term,Math.log((N+1)/(df+1))+1);chs.forEach(c=>{const tf=new Map();const toks=tokens(c.text).map(stem);toks.forEach(t=>tf.set(t,(tf.get(t)||0)+1));c.vec=new Map();for(const [t,f] of tf){c.vec.set(t,(f/toks.length)*(idf.get(t)||0));}});return {chs,idf};}function cos(a,b){let d=0,na=0,nb=0;a.forEach((va,t)=>{const vb=b.get(t)||0;d+=va*vb;na+=va*va});b.forEach(vb=>nb+=vb*vb);return (na&&nb)?(d/(Math.sqrt(na)*Math.sqrt(nb))):0}function vec(idf,terms){const tf=new Map();terms.forEach(t=>tf.set(t,(tf.get(t)||0)+1));const v=new Map();const n=Math.max(1,terms.length);terms.forEach(t=>v.set(t,(tf.get(t)/n)*(idf.get(t)||0.5)));return v}const boot=window.BOOT;const {chs,idf}=build(boot.docs);function expand(q){const ts=tokens(q).map(stem);const out=new Set(ts);ts.forEach(t=>{if(t.endsWith('s')) out.add(t.replace(/s$/,'')); else out.add(t+'s'); out.add(strip(t));});return Array.from(out);}function search(q,k=6,thr=0.12){const ex=expand(q);const qv=vec(idf,ex);const sc=[];chs.forEach(c=>{const s=cos(qv,c.vec);if(s>=thr) sc.push({s,c})});sc.sort((a,b)=>b.s-a.s);if(sc.length) return sc.slice(0,k);return chs.slice(0,Math.min(k,6)).map(c=>({s:0.01,c}))}const log=document.getElementById("log");const push=(role,text)=>{const b=document.createElement("div");b.className="bubble "+(role==="user"?"user":"bot");b.textContent=text;log.appendChild(b);log.scrollTop=log.scrollHeight};function compose(q,h){const sents=[];const seen=new Set();h.forEach(x=>x.c.text.split(/(?<=[\\.\\!\\?])\\s+/).forEach(y=>{const t=y.trim();if(t.length<40)return;const k=t.toLowerCase();if(seen.has(k))return;seen.add(k);sents.push({t,sc:x.s});}));sents.sort((a,b)=>b.sc-a.sc);const pick=sents.slice(0,6).map(x=>'• '+x.t).join('\\n');return pick || "Esto es lo que puedo extraer relevante:\\n"+chs.slice(0,5).map(c=>'• '+c.text.slice(0,160)+'…').join('\\n');}function handle(){const i=document.getElementById("ask");const q=i.value.trim();if(!q)return;i.value="";push("user",q);const m=(boot.qa||[]).reduce((best,cur)=>{try{const vA=vec(idf,expand(q)), vB=vec(idf,expand(cur.q||""));const s=cos(vA,vB);return s>(best.s||0)?{s,cur}:best;}catch{return best;}},{s:0});if(m.s>=0.30){push("bot",m.cur.a+(m.cur.src?("\\n\\nFuente: "+m.cur.src):""));return}const hits=search(q,6,0.12);push("bot",compose(q,hits));}document.getElementById("send").addEventListener("click",handle);document.getElementById("ask").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();handle()}});})();</script></body></html>`;
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (state.bot.name ? state.bot.name.toLowerCase().replace(/\s+/g,"-") : "asistente") + "-static.html";
  document.body.appendChild(a); a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1500);
  a.remove();
}

/* ===================== Wizard JSONL + Autocompletar ===================== */
function msg(t){ alert(t); }
function fillMerge(curr, add){ if (!add) return curr||""; if (!curr) return add; if (curr.includes(add)) return curr; return curr+"\n\n"+add; }
function extractFromText(url, text){
  const lines = (text||"").split(/\n+/).map(s=>s.trim()).filter(Boolean);
  let name=""; try{ const h=new URL(url).hostname.replace(/^www\./,''); const base=h.split('.')[0]; name=base?base.charAt(0).toUpperCase()+base.slice(1):""; }catch{}

  const emails = Array.from((text||"").matchAll(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi)).map(m=>m[0]);
  const phones = Array.from((text||"").matchAll(/(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,3}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}/g)).map(m=>m[0]).filter(x=>x.replace(/\D/g,'').length>=7);

  const hourLines = lines.filter(l=>/(horario|lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|\b\d{1,2}:\d{2}\b|\bam\b|\bpm\b)/i.test(l)).slice(0,8);
  const offerLines = lines.filter(l=>/(?:\$|\bUSD\b|\bCOP\b|\bMXN\b|\bS\/\b|\bAR\$)|\bprecio|\bplan|\bservicio|\bpaquete/i.test(l)).slice(0,12);
  const policyLines = lines.filter(l=>/(pol[ií]tica|t[eé]rminos|condiciones|garant[ií]a|devoluci[oó]n|cambios|privacidad)/i.test(l)).slice(0,12);

  const faqs = [];
  for (let i=0;i<lines.length;i++){
    const L = lines[i];
    if (/\?/.test(L)) {
      const q = L.replace(/\s+/g," ").trim();
      const a = (lines[i+1]||"").replace(/\s+/g," ").trim();
      if (q && a && !/\?$/.test(a)) { faqs.push({ q, a }); if (faqs.length>=10) break; }
    }
  }

  let desc=""; const aboutIdx = lines.findIndex(l=>/(sobre|qu[ií]enes somos|qu[eé] es|nosotros|acerca)/i.test(l));
  const pick = aboutIdx>=0 ? lines.slice(aboutIdx, aboutIdx+6) : lines.slice(0,6);
  desc = pick.join(" ").replace(/\s+/g," ").trim().slice(0,700);

  const contactParts = [];
  if (emails.length) contactParts.push(`Emails: ${[...new Set(emails)].join(", ")}`);
  if (phones.length) contactParts.push(`Teléfonos: ${[...new Set(phones)].join(", ")}`);
  const contactExtras = lines.filter(l=>/(whats?app|contacto|direcci[oó]n|ubicaci[oó]n|atenci[oó]n|soporte)/i.test(l)).slice(0,5);
  const contact = fillMerge(contactParts.join("\n"), contactExtras.join("\n"));

  return { name, desc, contact, hours:hourLines.join("\n"), offers:offerLines.join("\n"), policies:policyLines.join("\n"), faqs };
}
async function autoFillFromUrl(url){
  if (!url) throw new Error("Pega una URL primero.");
  const txt = await fetchUrlText(url);
  if (!txt || txt.length<40) throw new Error("No pude leer esa URL.");
  const data = extractFromText(url, txt);
  if (data.name && !$("w_name").value) $("w_name").value = data.name;
  $("w_desc").value     = fillMerge($("w_desc").value, data.desc);
  $("w_contact").value  = fillMerge($("w_contact").value, data.contact);
  $("w_hours").value    = fillMerge($("w_hours").value, data.hours);
  $("w_offers").value   = fillMerge($("w_offers").value, data.offers);
  $("w_policies").value = fillMerge($("w_policies").value, data.policies);
  if (data.faqs?.length){
    const lines = data.faqs.map(f => `${f.q} | ${f.a}`);
    $("w_faqs").value = fillMerge($("w_faqs").value, lines.join("\n"));
  }
}
function parseFaqLine(line){
  const raw = line.trim(); if (!raw) return null;
  const parts = raw.split(/\s*\|\s*/); if (parts.length<2) return null;
  let q=(parts[0]||"").trim(); let a=(parts.slice(1).join(" | ")||"").trim();
  if (!q||!a) return null; const hadQ=/[¿?]/.test(q); q=q.replace(/\?+$/,''); if (!/\?$/.test(q)) q=q+(hadQ?"":"?");
  return { q, a, src:"faq" };
}
function pairsFromProfile(){
  const goal = (state.bot.goal||"").trim();
  const notes = (state.bot.notes||"").trim();
  const pairs = [];
  if (goal) pairs.push({ q:"¿Cuál es el objetivo de este asistente?", a:goal, src:"perfil" });
  if (notes) pairs.push({ q:"¿Qué debo saber para atender bien al cliente?", a:notes, src:"perfil" });
  if (!goal && !notes){
    pairs.push({ q:"¿Cómo respondes?", a:"De forma clara y útil, pidiendo los datos mínimos y dando siguientes pasos concretos.", src:"perfil" });
  }
  return pairs;
}
function pairsFromWizard(){
  const name = ($("w_name")?.value||"").trim();
  const tone = ($("w_tone")?.value||"").trim() || "Cercano y profesional.";
  const desc = ($("w_desc")?.value||"").trim();
  const contact = ($("w_contact")?.value||"").trim();
  const hours = ($("w_hours")?.value||"").trim();
  const offersLines = ($("w_offers")?.value||"").trim().split(/\r?\n/).filter(Boolean);
  const policies = ($("w_policies")?.value||"").trim();
  const faqsLines = ($("w_faqs")?.value||"").trim().split(/\r?\n/).filter(Boolean);

  const pairs = [];
  if (name || desc){
    pairs.push({ q:`¿Qué es ${name||"esta empresa"} y qué hace?`, a:`${desc||"Empresa que ayuda a clientes con productos/servicios."}\n\nTono: ${tone}`, src:"perfil" });
  } else {
    pairs.push({ q:"¿Qué hace esta empresa?", a:"Ayuda a clientes con sus necesidades. Puedo orientarte en precios, procesos y soporte.", src:"perfil" });
  }
  if (contact) pairs.push({ q:"¿Cómo puedo contactarlos?", a:contact, src:"contacto" });
  if (hours)   pairs.push({ q:"¿Cuáles son los horarios y cobertura?", a:hours, src:"operación" });
  if (offersLines.length){
    pairs.push({ q:"¿Qué productos/servicios ofrecen y precios?", a:offersLines.map((l,i)=>`${i+1}. ${l}`).join("\n"), src:"oferta" });
  }
  if (policies) pairs.push({ q:"¿Cuáles son las políticas de garantía, cambios y tiempos?", a:policies, src:"políticas" });
  faqsLines.forEach(line=>{ const p=parseFaqLine(line); if (p) pairs.push(p); });
  if (!pairs.length) pairs.push(...pairsFromProfile());
  return pairs;
}
function jsonlString(pairs){ return pairs.map(p=>JSON.stringify(p)).join("\n"); }
function downloadJsonl(pairs, name="dataset_chatbot.jsonl"){
  if (!pairs.length){ msg("No hay pares para descargar."); return; }
  const blob = new Blob([jsonlString(pairs)], {type:"application/jsonl;charset=utf-8"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(()=>URL.revokeObjectURL(a.href), 1200); a.remove();
}
function addPairsToProject(pairs){
  if (!pairs.length){ msg("No hay pares para agregar."); return; }
  state.qa.push(...pairs);
  const txt = pairs.map(x=>`PREGUNTA: ${x.q}\nRESPUESTA: ${x.a}${x.src?`\nFUENTE: ${x.src}`:""}`).join("\n\n");
  const sid = nowId();
  state.sources.push({id:sid, type:'file', title:'dataset_chatbot (wizard).jsonl', addedAt:Date.now()});
  state.docs.push({id:nowId(), sourceId:sid, title:'dataset_chatbot (wizard).jsonl', text:txt, chunks:[]});
  buildIndex(); save(); renderSources();
  $("modelStatus").textContent = "Con conocimiento";
  msg(`Se agregaron ${pairs.length} pares al proyecto.`);
}
function readPairsFromPreview(){
  const raw = ($("w_preview")?.value||"").trim(); if (!raw) return [];
  const lines = raw.split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
  const pairs = [];
  for (let i=0;i<lines.length;i++){
    try{
      const obj = JSON.parse(lines[i]);
      if (obj && obj.q && obj.a){ pairs.push({ q:String(obj.q), a:String(obj.a), src: obj.src?String(obj.src):undefined }); }
      else throw new Error("Falta q/a");
    }catch(e){ throw new Error(`Línea ${i+1} inválida: ${e.message}`); }
  }
  return pairs;
}
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
    msg(`Generado: ${pairs.length} pares (vista previa).`);
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
    }catch(e){ msg(e.message || "No se pudo agregar."); }
  });
  btnFromProfile?.addEventListener("click", ()=>{
    const pairs = pairsFromProfile();
    previewArea.value = jsonlString(pairs);
    msg(`Generado desde Objetivo/Notas: ${pairs.length} pares.`);
  });

  // Autocompletar desde URL
  const urlInput = $("w_url");
  const btnAuto = $("wizAutofill");
  btnAuto?.addEventListener("click", async ()=>{
    const url = urlInput?.value.trim();
    if (!url) return msg("Pega una URL válida.");
    btnAuto.disabled = true; const old = btnAuto.textContent; btnAuto.textContent = "Cargando…";
    try{
      await autoFillFromUrl(url);
      msg("Campos sugeridos desde la URL. Revisa y ajusta lo necesario.");
    }catch(e){ msg(e.message || "No pude autocompletar desde esa URL."); }
    finally{ btnAuto.disabled = false; btnAuto.textContent = old || "Autocompletar"; }
  });

  modal.addEventListener("click", (e)=>{ if (e.target === modal) modal.classList.remove("show"); });
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
  const list = $("sourcesList"); list.innerHTML="";
  if (!state.sources.length){ list.appendChild(el("div",{class:"muted small", text:"Aún no has cargado fuentes."})); return; }
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
  const list = $("corpusList"); list.innerHTML="";
  if (!state.docs.length){ list.appendChild(el("div",{class:"muted small", text:"Sin documentos. Sube archivos o añade URLs."})); return; }
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
  const list = $("urlList"); list.innerHTML="";
  if (!state.urlsQueue.length){ list.appendChild(el("div",{class:"muted small", text:"No hay URLs en cola."})); return; }
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
  const log = $("chatlog"); log.innerHTML="";
  state.chat.forEach(m=>{
    const b = el("div",{class:`bubble ${m.role==='user'?'user':'bot'}`}); b.textContent = m.text; log.appendChild(b);
  });
  log.scrollTop = log.scrollHeight;
}
function renderMiniChat(){
  const log = $("miniLog"); log.innerHTML="";
  state.miniChat.forEach(m=>{
    const b = el("div",{class:`bubble ${m.role==='user'?'user':'bot'}`}); b.textContent = m.text; log.appendChild(b);
  });
  log.scrollTop = log.scrollHeight;
}
function setBusy(flag){
  ingestBusy = flag;
  ["btnIngestFiles","btnCrawl","btnTrain","btnRebuild","btnReset"].forEach(id=>{ if ($(id)) $(id).disabled = flag; });
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

  $("btnTrain").addEventListener("click", ()=>{ buildIndex(); $("modelStatus").textContent="Con conocimiento"; alert("Entrenamiento (índice + análisis) listo."); });
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
      $("filePicker").value = "";
    }
  });

  $("btnAddUrl").addEventListener("click", ()=>{
    const url = $("urlInput").value.trim(); if (!url) return;
    state.urlsQueue.push({id:nowId(), url, title:""}); $("urlInput").value = ""; save(); renderUrlQueue();
  });
  $("btnCrawl").addEventListener("click", async ()=>{
    if (!state.urlsQueue.length) return alert("Añade al menos una URL.");
    await ingestUrls(state.urlsQueue); state.urlsQueue = []; save(); renderUrlQueue();
  });
  $("btnClearSources").addEventListener("click", ()=>{ state.urlsQueue = []; save(); renderUrlQueue(); });

  $("btnSearchCorpus").addEventListener("click", ()=>{
    const q = $("searchCorpus").value.trim(); if (!q) return;
    const hits = searchChunks(q, state.bot.topk, state.bot.threshold);
    const list = $("corpusList"); list.innerHTML="";
    if (!hits.length){ list.appendChild(el("div",{class:"muted small", text:"Sin coincidencias."})); return; }
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

  $("btnRebuild").addEventListener("click", ()=>{ state.docs.forEach(d=> d.chunks=[]); buildIndex(); save(); alert("Reconstruido el índice + análisis."); });
  $("btnReset").addEventListener("click", ()=>{
    if (!confirm("Esto borrará todo el conocimiento y configuración guardada. ¿Continuar?")) return;
    state.sources=[]; state.docs=[]; state.index={vocab:new Map(), idf:new Map(), built:false};
    state.urlsQueue=[]; state.chat=[]; state.miniChat=[]; state.qa=[]; state.kb={contact:[],hours:[],prices:[],policies:[],products:[],services:[],locations:[],faqs:[],sentences:[],keyphrases:new Map()};
    state.settings = { allowWeb:true, strictContext:true };
    $("ingestProgress").style.width="0%";
    save(); renderSources(); renderCorpus(); renderUrlQueue(); renderChat(); renderMiniChat();
    $("modelStatus").textContent = "Sin entrenar";
  });

  if ($("allowWeb")) $("allowWeb").addEventListener("change", e=>{ state.settings.allowWeb = !!e.target.checked; save(); });
  if ($("strictContext")) $("strictContext").addEventListener("change", e=>{ state.settings.strictContext = !!e.target.checked; save(); });

  if ($("btnExportHtml")) $("btnExportHtml").addEventListener("click", exportStandaloneHtml);

  if ($("send")) $("send").addEventListener("click", ()=> handleAsk("ask","tester"));
  if ($("ask")) $("ask").addEventListener("keydown", (e)=>{ if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); handleAsk("ask","tester"); } });
  if ($("launcher")) $("launcher").addEventListener("click", ()=>{ $("mini").classList.add("show"); });
  if ($("closeMini")) $("closeMini").addEventListener("click", ()=>{ $("mini").classList.remove("show"); });
  if ($("miniSend")) $("miniSend").addEventListener("click", ()=> handleAsk("miniAsk","mini"));
  if ($("miniAsk")) $("miniAsk").addEventListener("keydown", (e)=>{ if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); handleAsk("miniAsk","mini"); } });

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

  // 1) Q&A directas
  let bestQA = null, bestScore = 0;
  if (state.qa.length){
    const qv = vectorizeQuery(q);
    for (let i=0;i<state.qa.length;i++){
      const s = cosineSim(qv, vectorizeQuery(state.qa[i].q));
      if (s>bestScore){ bestScore=s; bestQA = state.qa[i]; }
    }
  }
  if (bestQA && bestScore>=0.30){
    pushAndRender(scope, 'assistant', bestQA.a + (bestQA.src?`\n\nFuente: ${bestQA.src}`:""));
    return;
  }

  // 2) Búsqueda híbrida con degradación
  const hits = searchChunks(q, state.bot.topk, state.bot.threshold);

  // 3) Composición sin silencios
  let draft = composeAnswer(q, hits);

  // 4) Pulir con backend IA (opcional, nunca deja vacío)
  if (AI_SERVER_URL){
    askServerAI(q, scope, draft).then(ai=>{
      pushAndRender(scope,'assistant', (ai||draft));
    });
    return;
  }

  // 5) Sin backend: responder con el borrador sí o sí
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
