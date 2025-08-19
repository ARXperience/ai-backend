// bot.js — PRO + Flujo de Cotización + Chips extra + Voz (STT/TTS) + Markdown + Copiar + Persistencia

/***** 1) Referencias DOM *****/
const msgs   = document.getElementById('messages');
const input  = document.getElementById('input');
const send   = document.getElementById('send');
const typing = document.getElementById('typing');
const clear  = document.getElementById('clear');

// Botones de voz (barra superior y junto al input)
const micToggle   = document.getElementById('micToggle');
const ttsToggle   = document.getElementById('ttsToggle');
const micComposer = document.getElementById('micComposer');

/***** 2) Constantes de almacenamiento *****/
const STORAGE_KEY = 'cdd_chat_history_v1';
const QUOTE_KEY   = 'cdd_quote_leads_v1';
const FLOW_KEY    = 'cdd_quote_flow_state_v1';
const TTS_KEY     = 'cdd_tts_enabled';

/***** 3) Config: contacto oficial *****/
const OFICIAL_PHONE = "573028618806";
const OFICIAL_MAIL  = "centrodigitaldediseno@gmail.com";

/***** 4) Estado global seguro (ANTES de cualquier render) *****/
let ttsEnabled = false;  // se actualiza abajo desde localStorage
try { ttsEnabled = JSON.parse(localStorage.getItem(TTS_KEY) || 'false'); } catch { ttsEnabled = false; }

let flow = loadFlowState() || {
  activo: false, paso: 0, datos: { nombre:"", servicios:"", empresa:"", telefono:"" }
};

/***** 5) Base de conocimiento *****/
const CTA = `\n\n**Contáctanos:** WhatsApp +${OFICIAL_PHONE} · ${OFICIAL_MAIL}`;
const KB = {
  servicios:
    "### ¿Qué hacemos?\n- **Páginas web con IA** (landing, multipágina, e-commerce).\n- **Branding** (identidad visual).\n- **Contenido para redes** (Reels/TikTok/Shorts) con **SEO social**.\n- **Automatizaciones** (ManyChat, Make, WhatsApp Business API).\n- **Bots de IA** para atención y ventas 24/7.\n- **E-commerce & embudos** (checkout, analítica).\n- **Creativos con IA** (foto/video de producto, anuncios).\n- **AR** y **Ads (Meta)**.\n- **Growth Partner** enfocado en KPIs." + CTA,

  web:
    "### Páginas web & tiendas\nDiseño moderno, rápido y orientado a conversión (estructura, copy, analítica).\nIntegración con **WhatsApp/CRM/ManyChat** para capturar y nutrir leads." + CTA,

  automat:
    "### Automatizaciones & Bots de IA\n- **ManyChat/WhatsApp**: flujos, segmentación, campañas.\n- **Make**: integra formularios, CRMs, Google, Email, Meta, etc.\n- **Bots de IA** entrenados con tus textos/FAQs para calificar leads y derivar a humano." + CTA,

  cotiz:
    "### Precios & cotización\nTrabajamos **por alcance y objetivos**; el valor depende de páginas, integraciones, volumen de contenido y automatizaciones.\n> Nota: las **apps premium** no son gratuitas (costo mensual del proveedor).\n\n**Cómo cotizamos**\n1) **Brief** rápido + **llamada** de 15–20 min.\n2) Propuesta con **entregables, tiempos y valor**.\n3) Alineación y arranque del **Sprint 1**." + CTA,

  agente:
    `### Crea tu **agente gratis**
Lanza un prototipo en minutos y pruébalo en tu web o WhatsApp.

<a href="https://gold-snail-248674.hostingersite.com/chatbot.html" target="_blank" style="display:inline-block;margin-top:8px;margin-right:8px;background:#10a37f;color:#fff;text-decoration:none;padding:10px 16px;border-radius:12px;font-weight:700">🚀 Crear agente gratis</a>`
};

/***** 6) Arranque *****/
restoreHistory();
if (historyEmpty()) {
  botMsg("👋 **Hola, soy el asistente del Centro Digital de Diseño.**\nRespondo sobre **servicios**, **páginas web**, **automatizaciones** y **cotización**.");
}

/***** 7) Listeners UI *****/
send?.addEventListener('click', () => {
  const txt = input.value.trim();
  if (!txt) return;
  input.value = "";
  userMsg(txt);
  route(txt);
});
input?.addEventListener('keydown', e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send?.click(); }
});
document.querySelectorAll(".chip").forEach(c => {
  c.addEventListener('click', () => { userMsg(c.dataset.q); route(c.dataset.q); });
});
clear?.addEventListener('click', () => {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(FLOW_KEY);
  msgs.innerHTML = "";
  typing.style.display="none";
  flow = { activo:false, paso:0, datos:{nombre:"",servicios:"",empresa:"",telefono:""} };
  botMsg("🧹 Historial limpio. ¿Quieres **cotizar**? Puedo guiarte paso a paso.");
});

/***** 8) Router principal *****/
function route(q){
  if (/^cancelar$/i.test(q.trim())) {
    if (flow.activo){
      flow = { activo:false, paso:0, datos:{nombre:"",servicios:"",empresa:"",telefono:""} };
      saveFlowState();
      return botMsg("Flujo de cotización **cancelado**. Cuando quieras, escribe *cotizar* para retomarlo.");
    }
  }
  if (flow.activo) { handleCotizacion(q); return; }

  const qn = norm(q);
  if (/(cotiz|presupuesto|precio|cu[aá]nto vale|cu[aá]nto cuesta)/.test(qn)) {
    startCotizacion(); return;
  }
  respond(q);
}

/***** 9) Flujo de Cotización *****/
function startCotizacion(){
  flow = { activo:true, paso:1, datos:{ nombre:"", servicios:"", empresa:"", telefono:"" } };
  saveFlowState();
  botMsg("¡Perfecto! Para darte una **cotización personalizada** necesito unos datos.\n\n1️⃣ ¿Cuál es tu **nombre completo**?\n\n*(Puedes escribir `cancelar` para salir del flujo.)*");
}

function handleCotizacion(respuesta){
  const text = respuesta.trim();
  switch(flow.paso){
    case 1: {
      flow.datos.nombre = text;
      flow.paso = 2; saveFlowState();
      botMsg(`Gracias, **${escapeHTML(text)}**.  
2️⃣ Cuéntame: ¿Qué **servicios** te interesan?  
_Ej.: “Landing page + automatización WhatsApp”, “E-commerce con branding”, “Bot de IA para atención”, etc._`);
      break;
    }
    case 2: {
      flow.datos.servicios = text;
      flow.paso = 3; saveFlowState();
      botMsg("3️⃣ ¿Cómo se llama tu **empresa o proyecto**?");
      break;
    }
    case 3: {
      flow.datos.empresa = text;
      flow.paso = 4; saveFlowState();
      botMsg("4️⃣ ¿Cuál es tu **número de WhatsApp o teléfono** para compartirte la propuesta?");
      break;
    }
    case 4: {
      if (!isValidPhone(text)) {
        botMsg("Parece que el número no es válido. Intenta con un formato como `3001234567` o incluye código de país `+57 3001234567`.");
        return;
      }
      flow.datos.telefono = cleanPhone(text);
      finalizeQuote();
      break;
    }
    default:
      flow = { activo:false, paso:0, datos:{nombre:"",servicios:"",empresa:"",telefono:""} };
      saveFlowState();
      botMsg("He reiniciado el flujo. Escribe **cotizar** para empezar de nuevo.");
  }
}

function finalizeQuote(){
  const leads = JSON.parse(localStorage.getItem(QUOTE_KEY) || "[]");
  const lead = { ...flow.datos, fecha: new Date().toISOString() };
  leads.push(lead);
  localStorage.setItem(QUOTE_KEY, JSON.stringify(leads));

  const { nombre, servicios, empresa, telefono } = flow.datos;
  const wappText = encodeURIComponent(
    `Hola, soy ${nombre} (${empresa}). Me interesa: ${servicios}. Mi contacto: ${telefono}.`
  );
  const mailBody = encodeURIComponent(
`Nombre: ${nombre}
Servicios: ${servicios}
Empresa/Proyecto: ${empresa}
Teléfono: ${telefono}

Mensaje: Hola, quiero avanzar con la cotización.`
  );

  const btnStyle = "display:inline-block;margin-top:8px;margin-right:8px;background:#10a37f;color:#fff;text-decoration:none;padding:8px 14px;border-radius:10px;font-weight:600;font-size:14px";

  const resumen =
`### ¡Genial, ${escapeHTML(nombre)}! 🙌
Con estos datos armamos tu propuesta con **entregables, tiempos y valor**. Te contactaremos en breve.

**Resumen**
- **Servicios:** ${escapeHTML(servicios)}
- **Empresa/Proyecto:** ${escapeHTML(empresa)}
- **WhatsApp/Teléfono:** ${escapeHTML(telefono)}

**Acceso rápido**  
<a href="https://wa.me/${OFICIAL_PHONE}?text=${wappText}" target="_blank" style="${btnStyle}">📲 WhatsApp Oficial</a>
<a href="mailto:${OFICIAL_MAIL}?subject=Cotización&body=${mailBody}" style="${btnStyle}">✉️ Email Oficial</a>

> Si necesitas corregir algo, escribe **cotizar** para iniciar nuevamente.`;

  flow = { activo:false, paso:0, datos:{nombre:"",servicios:"",empresa:"",telefono:""} };
  saveFlowState();
  botMsg(resumen);
  botMsg(KB.cotiz);
}

/***** 10) Respuestas estándar *****/
function respond(q){
  showTyping(true);
  setTimeout(() => {
    showTyping(false);
    const qn = norm(q);

    if ( /(servicios|qué hacen|que hacen|ofrecen|todo lo que hacen)/.test(qn) ) return botMsg(KB.servicios);
    if ( /(web|landing|tienda|ecommerce|shopify|woocommerce|página|pagina)/.test(qn) ) return botMsg(KB.web);
    if ( /(automat|whatsapp|manychat|make|bot|ia|integraci[oó]n|crm)/.test(qn) ) return botMsg(KB.automat);
    if ( /(precio|cu[aá]nto vale|cu[aá]nto cuesta|cotizaci[oó]n|presupuesto|cotizar)/.test(qn) ) {
      botMsg(KB.cotiz + "\n\n¿Quieres que **inicie el flujo de cotización** aquí mismo? Escribe **cotizar**.");
      return;
    }
    if ( /(agente gratis|crear un agente gratis|crear agente)/.test(qn) ) {
      return botMsg(KB.agente);
    }
    if ( /(agendar llamada|agenda llamada|quiero llamada|hablar por whatsapp)/.test(qn) ) {
      const link = `https://wa.me/${OFICIAL_PHONE}?text=${encodeURIComponent("Hola, quiero agendar una llamada para hablar del proyecto.")}`;
      const btn  = `<a href="${link}" target="_blank" style="display:inline-block;margin-top:8px;margin-right:8px;background:#10a37f;color:#fff;text-decoration:none;padding:10px 16px;border-radius:12px;font-weight:700">📲 Agendar por WhatsApp</a>`;
      return botMsg(`### Agendar llamada\nCoordinemos por WhatsApp con nuestro equipo.\n\n${btn}`);
    }

    // Buscador difuso simple
    const hit = smallSearch(qn);
    if (hit) return botMsg(hit);

    botMsg("Puedo ayudarte con **servicios**, **páginas web**, **automatizaciones** y **cotización**. ¿Qué necesitas exactamente?\n\nEj.: *“Landing + WhatsApp”*, *“Calendarizar contenido con IA”*." + CTA);
  }, 420 + Math.random()*260);
}

function smallSearch(q){
  const pairs = [
    [KB.servicios, ["branding","seo social","growth","ads","ar","reels","tiktok","shorts","contenido"]],
    [KB.web,       ["web","landing","tienda","ecommerce","shopify","woocommerce","velocidad","conversion","analitica","analytics","crm","whatsapp"]],
    [KB.automat,   ["automat","manychat","whatsapp","make","bot","flow","crm","integracion","integración"]],
    [KB.cotiz,     ["precio","cotiz","presupuesto","propuesta","valor"]],
    [KB.agente,    ["agente","gratis","chatbot","crear","bot"]]
  ];
  let best=null,score=0;
  pairs.forEach(([text,keys])=>{
    const s = keys.reduce((acc,k)=> acc + (q.includes(k)?1:0), 0);
    if (s>score){score=s; best=text;}
  });
  return score>0 ? best : null;
}

/***** 11) Render + Markdown + Copiar + Autoscroll + TTS *****/
function render(role, mdText){
  const row = document.createElement("div");
  row.className = "row " + (role === "assistant" ? "assistant" : "user");

  const av = document.createElement("div");
  av.className = "avatar";
  av.textContent = role === "assistant" ? "AI" : "Tú";

  const bub = document.createElement("div");
  bub.className = "bubble";

  let html = mdToHTML(mdText);
  // Botónizar auto enlaces "WhatsApp:" y "Email:"
  html = html
    .replace(/WhatsApp:\s*(https?:\/\/[^\s<]+)/gi, (_m, url) => {
      const btnStyle = "display:inline-block;margin-top:8px;margin-right:8px;background:#10a37f;color:#fff;text-decoration:none;padding:8px 14px;border-radius:10px;font-weight:600;font-size:14px";
      return `<a href="${url}" target="_blank" style="${btnStyle}">📲 WhatsApp</a>`;
    })
    .replace(/Email:\s*(mailto:[^\s<]+)/gi, (_m, url) => {
      const btnStyle = "display:inline-block;margin-top:8px;margin-right:8px;background:#10a37f;color:#fff;text-decoration:none;padding:8px 14px;border-radius:10px;font-weight:600;font-size:14px";
      return `<a href="${url}" style="${btnStyle}">✉️ Email</a>`;
    });

  bub.innerHTML = html;

  // Copiar en bloques <pre>
  bub.querySelectorAll("pre").forEach(pre => {
    const head = document.createElement("div");
    head.className = "code-head";
    head.innerHTML = `<span>código</span>`;
    const btn = document.createElement("button");
    btn.className = "copy";
    btn.textContent = "Copiar";
    btn.addEventListener("click", ()=>{
      const code = pre.querySelector("code")?.innerText || pre.innerText;
      navigator.clipboard.writeText(code);
      btn.textContent = "Copiado ✓";
      setTimeout(()=> btn.textContent = "Copiar", 1100);
    });
    pre.parentNode.insertBefore(head, pre);
    head.appendChild(btn);
  });

  row.appendChild(av);
  row.appendChild(bub);
  msgs.appendChild(row);

  requestAnimationFrame(() => { msgs.scrollTop = msgs.scrollHeight; });

  saveToHistory(role, mdText);

  // TTS solo para respuestas del asistente
  if (role === "assistant" && ttsEnabled) {
    speakText(stripMarkdown(mdText));
  }
}
function userMsg(text){ render("user", escapeHTML(text)); }
function botMsg(text){ render("assistant", text); }

function showTyping(v){ typing.style.display = v ? "flex" : "none"; }

/***** 12) Markdown mínimo *****/
function mdToHTML(md){
  md = md.replace(/```([\s\S]*?)```/g, (_,code)=> `<pre><code>${escapeHTML(code.trim())}</code></pre>`);
  md = md
    .replace(/^### (.*)$/gim,'<h3>$1</h3>')
    .replace(/^## (.*)$/gim,'<h2>$1</h2>')
    .replace(/^# (.*)$/gim,'<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/`([^`]+?)`/g,'<code>$1</code>');
  const lines = md.split('\n').map(line=>{
    if (/^\s*-\s+/.test(line)) return `<li>${line.replace(/^\s*-\s+/, '')}</li>`;
    if (/^\s*•\s+/.test(line)) return `<li>${line.replace(/^\s*•\s+/, '')}</li>`;
    if (/^<h\d|^<pre|^<ul|^<li|^<\/li|^<\/ul|^<a /.test(line)) return line;
    return line.trim()? `<p>${line}</p>` : '<p style="margin:4px 0"></p>';
  });
  const joined = lines.join('\n').replace(/(?:<li>[\s\S]*?<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
  return joined;
}
function stripMarkdown(s){
  return s
    .replace(/```[\s\S]*?```/g,' ')
    .replace(/[*_#>`~-]/g,' ')
    .replace(/\[(.*?)\]\((.*?)\)/g,'$1')
    .replace(/\s+/g,' ')
    .trim();
}
function escapeHTML(s){return (s||'').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));}
function norm(s){return (s||'').toLowerCase()
  .normalize('NFD').replace(/\p{Diacritic}/gu,'')
  .replace(/[^a-z0-9áéíóúñü\s]/g,' ')
  .replace(/\s+/g,' ')
  .trim();
}

/***** 13) Validaciones *****/
function isValidPhone(v){
  const d = onlyDigits(v);
  if (/^57\d{10}$/.test(d)) return true;
  if (/^\d{10}$/.test(d))   return true;
  return false;
}
function cleanPhone(v){
  let d = onlyDigits(v);
  if (/^\d{10}$/.test(d)) d = "57" + d;
  return d;
}
function onlyDigits(s){ return (s||'').replace(/\D+/g,''); }

/***** 14) Persistencia chat/flujo *****/
function saveToHistory(role, text){
  const arr = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  arr.push({ role, text, t: Date.now() });
  localStorage.setItem(STORAGE_KEY, JSON.stringify(arr));
}
function restoreHistory(){
  const arr = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  if (!arr.length) return;
  arr.forEach(m => { m.role === 'assistant' ? botMsg(m.text) : userMsg(m.text); });
  const savedFlow = loadFlowState();
  if (savedFlow?.activo){
    flow = savedFlow;
    botMsg("Teníamos un **flujo de cotización** pendiente. ¿Deseas **continuar**? Si prefieres salir, escribe `cancelar`.");
  }
}
function historyEmpty(){
  const arr = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  return arr.length === 0;
}
function saveFlowState(){
  localStorage.setItem(FLOW_KEY, JSON.stringify(flow));
}
function loadFlowState(){
  try { return JSON.parse(localStorage.getItem(FLOW_KEY) || "null"); }
  catch { return null; }
}

/***** 15) VOZ: STT (Reconocimiento) y TTS (Síntesis) *****/
// STT
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;
let listening = false;

if (SR) {
  rec = new SR();
  rec.lang = 'es-CO';
  rec.continuous = false;
  rec.interimResults = true;

  rec.onresult = (e) => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++){
      const txt = e.results[i][0].transcript;
      if (e.results[i].isFinal){
        const finalTxt = txt.trim();
        if (finalTxt){
          input.value = finalTxt;
          send?.click();
        }
      } else {
        interim += txt + ' ';
      }
    }
    input.placeholder = interim ? `🎙️ ${interim.trim()}…` : 'Escribe tu mensaje…';
  };
  rec.onerror = () => { stopListening(); };
  rec.onend   = () => { stopListening(); };
}

function startListening(){
  if (!rec){
    botMsg("🎙️ Tu navegador no soporta entrada por voz. Usa **Chrome/Edge**.");
    return;
  }
  if (listening) return;
  try {
    rec.start();
    listening = true;
    setMicUI(true);
  } catch { /* noop */ }
}
function stopListening(){
  if (!rec) return;
  try { rec.stop(); } catch {}
  listening = false;
  setMicUI(false);
  input.placeholder = 'Escribe tu mensaje…';
}
function setMicUI(active){
  if (micToggle)   micToggle.textContent   = active ? '🛑 Detener' : '🎙️ Voz';
  if (micComposer) micComposer.textContent = active ? '🛑' : '🎙️';
}
micToggle?.addEventListener('click', () => listening ? stopListening() : startListening());
micComposer?.addEventListener('click', () => listening ? stopListening() : startListening());

// TTS
function updateTTSToggle(){
  if (ttsToggle) ttsToggle.textContent = ttsEnabled ? '🔇 Silenciar' : '🔊 Lectura';
}
updateTTSToggle();

ttsToggle?.addEventListener('click', () => {
  ttsEnabled = !ttsEnabled;
  localStorage.setItem(TTS_KEY, JSON.stringify(ttsEnabled));
  updateTTSToggle();
  if (!ttsEnabled && 'speechSynthesis' in window){
    window.speechSynthesis.cancel();
  }
});

function speakText(text){
  if (!('speechSynthesis' in window)) return;
  const msg = new SpeechSynthesisUtterance(text);
  // Intentar voz ES
  const pickVoice = () => {
    const voices = window.speechSynthesis.getVoices();
    return voices.find(v => /es[-_](CO|MX|ES)/i.test(v.lang)) || voices.find(v => /^es/i.test(v.lang));
  };
  let v = pickVoice();
  if (!v){
    window.speechSynthesis.onvoiceschanged = () => {
      v = pickVoice();
    };
  } else {
    msg.voice = v; msg.lang = v.lang;
  }
  msg.rate = 1.0; msg.pitch = 1.0; msg.volume = 1.0;
  try { window.speechSynthesis.cancel(); } catch {}
  window.speechSynthesis.speak(msg);
}
