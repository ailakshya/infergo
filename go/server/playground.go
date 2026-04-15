package server

import (
	"net/http"
)

// handlePlayground serves the embedded LLM playground UI.
func (s *Server) handlePlayground(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(playgroundHTML)) //nolint:errcheck
}

const playgroundHTML = `<!DOCTYPE html>
<html><head><title>infergo — Playground</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f0f1a;color:#e0e0e0;display:flex;flex-direction:column;height:100vh}
.header{padding:12px 20px;background:#16213e;border-bottom:1px solid #2a2a4a;display:flex;align-items:center;gap:12px}
.header h1{font-size:18px;font-weight:bold;color:#fff}
.header span{color:#0f0;font-size:12px}
.main{display:flex;flex:1;overflow:hidden}
.sidebar{width:280px;padding:16px;background:#12122a;border-right:1px solid #2a2a4a;display:flex;flex-direction:column;gap:16px;overflow-y:auto}
.sidebar label{font-size:13px;color:#999;font-weight:600;text-transform:uppercase;letter-spacing:0.5px}
.sidebar select,.sidebar textarea{width:100%;padding:8px 10px;border-radius:6px;border:1px solid #333;background:#1a1a2e;color:#eee;font-size:14px;font-family:inherit;resize:vertical}
.sidebar select{cursor:pointer}
.slider-row{display:flex;align-items:center;gap:8px}
.slider-row input[type=range]{flex:1;accent-color:#0f0}
.slider-val{font-size:13px;color:#0f0;min-width:32px;text-align:right}
.chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
.messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:85%;padding:12px 16px;border-radius:12px;line-height:1.6;white-space:pre-wrap;font-size:14px;word-break:break-word}
.msg.user{align-self:flex-end;background:#0a3d62;border-bottom-right-radius:4px}
.msg.assistant{align-self:flex-start;background:#2d2d44;border-bottom-left-radius:4px}
.input-bar{display:flex;padding:12px 20px;background:#16213e;border-top:1px solid #2a2a4a;gap:8px}
.input-bar textarea{flex:1;padding:10px 16px;border-radius:8px;border:1px solid #444;background:#1a1a2e;color:#eee;font-size:15px;font-family:inherit;resize:none;max-height:120px}
.input-bar button{padding:10px 24px;border-radius:8px;border:none;background:#0f0;color:#000;font-weight:bold;cursor:pointer;font-size:14px}
.input-bar button:hover{background:#0c0}
.input-bar button:disabled{background:#333;color:#666;cursor:not-allowed}
.field{display:flex;flex-direction:column;gap:6px}
</style></head><body>
<div class="header">
  <h1>infergo</h1><span>Playground</span>
</div>
<div class="main">
  <div class="sidebar">
    <div class="field">
      <label>Model</label>
      <select id="model"><option value="">Loading...</option></select>
    </div>
    <div class="field">
      <label>Temperature</label>
      <div class="slider-row">
        <input type="range" id="temp" min="0" max="2" step="0.05" value="0.7">
        <span class="slider-val" id="tempVal">0.70</span>
      </div>
    </div>
    <div class="field">
      <label>Max Tokens</label>
      <div class="slider-row">
        <input type="range" id="maxTok" min="16" max="4096" step="16" value="256">
        <span class="slider-val" id="maxTokVal">256</span>
      </div>
    </div>
    <div class="field">
      <label>System Prompt</label>
      <textarea id="sysPrompt" rows="5" placeholder="You are a helpful assistant."></textarea>
    </div>
  </div>
  <div class="chat-area">
    <div class="messages" id="messages"></div>
    <div class="input-bar">
      <textarea id="userInput" rows="1" placeholder="Type a message..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
      <button id="sendBtn" onclick="send()">Send</button>
    </div>
  </div>
</div>
<script>
const msgBox=document.getElementById('messages');
const userInput=document.getElementById('userInput');
const sendBtn=document.getElementById('sendBtn');
const modelSel=document.getElementById('model');
const tempSlider=document.getElementById('temp');
const tempVal=document.getElementById('tempVal');
const maxTokSlider=document.getElementById('maxTok');
const maxTokVal=document.getElementById('maxTokVal');
const sysPrompt=document.getElementById('sysPrompt');
let history=[];

tempSlider.oninput=function(){tempVal.textContent=parseFloat(this.value).toFixed(2)};
maxTokSlider.oninput=function(){maxTokVal.textContent=this.value};

async function loadModels(){
  try{
    const r=await fetch('/v1/models');
    const j=await r.json();
    modelSel.innerHTML='';
    if(j.data&&j.data.length>0){
      j.data.forEach(function(m){
        const o=document.createElement('option');
        o.value=m.id;o.textContent=m.id;
        modelSel.appendChild(o);
      });
    }else{
      const o=document.createElement('option');
      o.value='llm';o.textContent='llm (default)';
      modelSel.appendChild(o);
    }
  }catch(e){
    modelSel.innerHTML='<option value="llm">llm (default)</option>';
  }
}

function addMsg(role,text){
  const d=document.createElement('div');
  d.className='msg '+role;
  d.textContent=text;
  msgBox.appendChild(d);
  msgBox.scrollTop=msgBox.scrollHeight;
  return d;
}

async function send(){
  const text=userInput.value.trim();
  if(!text)return;
  userInput.value='';
  sendBtn.disabled=true;

  addMsg('user',text);
  history.push({role:'user',content:text});

  const msgs=[];
  const sys=sysPrompt.value.trim();
  if(sys)msgs.push({role:'system',content:sys});
  msgs.push.apply(msgs,history);

  const bot=addMsg('assistant','...');
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        model:modelSel.value,
        messages:msgs,
        max_tokens:parseInt(maxTokSlider.value),
        temperature:parseFloat(tempSlider.value),
        stream:true
      })});
    const reader=r.body.getReader();
    const dec=new TextDecoder();
    let out='';
    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      const lines=dec.decode(value).split('\n');
      for(const line of lines){
        if(!line.startsWith('data: ')||line==='data: [DONE]')continue;
        try{
          const j=JSON.parse(line.slice(6));
          const c=j.choices[0].delta.content;
          if(c){out+=c;bot.textContent=out;}
        }catch(e){}
      }
    }
    if(!out)bot.textContent='(no response)';
    history.push({role:'assistant',content:out||'(no response)'});
  }catch(e){
    bot.textContent='Error: '+e.message;
  }
  sendBtn.disabled=false;
  msgBox.scrollTop=msgBox.scrollHeight;
  userInput.focus();
}

loadModels();
</script></body></html>`
