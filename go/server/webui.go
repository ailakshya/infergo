package server

import (
	"net/http"
)

// handleWebUI serves a built-in chat interface.
func (s *Server) handleWebUI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(webUIHTML))
}

const webUIHTML = `<!DOCTYPE html>
<html><head><title>infergo</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui;background:#1a1a2e;color:#eee;display:flex;flex-direction:column;height:100vh}
.header{padding:12px 20px;background:#16213e;border-bottom:1px solid #333;font-size:18px;font-weight:bold}
.header span{color:#0f0;font-size:12px;margin-left:8px}
.chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:80%;padding:12px 16px;border-radius:12px;line-height:1.5;white-space:pre-wrap}
.user{align-self:flex-end;background:#0a3d62;border-bottom-right-radius:4px}
.bot{align-self:flex-start;background:#2d2d44;border-bottom-left-radius:4px}
.input{display:flex;padding:12px 20px;background:#16213e;border-top:1px solid #333;gap:8px}
input{flex:1;padding:10px 16px;border-radius:8px;border:1px solid #444;background:#1a1a2e;color:#eee;font-size:15px}
button{padding:10px 20px;border-radius:8px;border:none;background:#0f0;color:#000;font-weight:bold;cursor:pointer}
button:hover{background:#0c0}
</style></head><body>
<div class="header">infergo <span>v1.1.0</span></div>
<div class="chat" id="chat"></div>
<div class="input">
<input id="inp" placeholder="Type a message..." onkeydown="if(event.key==='Enter')send()">
<button onclick="send()">Send</button>
</div>
<script>
const chat=document.getElementById('chat'),inp=document.getElementById('inp');
async function send(){
  const text=inp.value.trim();if(!text)return;inp.value='';
  add('user',text);
  const bot=add('bot','...');
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'llm',messages:[{role:'user',content:text}],max_tokens:256,stream:true})});
    const reader=r.body.getReader();const dec=new TextDecoder();let out='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      const lines=dec.decode(value).split('\n');
      for(const line of lines){
        if(!line.startsWith('data: ')||line==='data: [DONE]')continue;
        try{const j=JSON.parse(line.slice(6));const c=j.choices[0].delta.content;
          if(c){out+=c;bot.textContent=out;}}catch{}
      }
    }
    if(!out)bot.textContent='(no response)';
  }catch(e){bot.textContent='Error: '+e.message;}
  chat.scrollTop=chat.scrollHeight;
}
function add(role,text){
  const d=document.createElement('div');
  d.className='msg '+(role==='user'?'user':'bot');
  d.textContent=text;chat.appendChild(d);
  chat.scrollTop=chat.scrollHeight;return d;
}
</script></body></html>`
