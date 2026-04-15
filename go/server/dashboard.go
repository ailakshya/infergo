package server

import (
	"net/http"
)

// handleDashboard serves the embedded health/metrics dashboard UI.
func (s *Server) handleDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(dashboardHTML)) //nolint:errcheck
}

const dashboardHTML = `<!DOCTYPE html>
<html><head><title>infergo — Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f0f1a;color:#e0e0e0;padding:20px}
.header{display:flex;align-items:center;gap:12px;margin-bottom:24px}
.header h1{font-size:22px;color:#fff}
.header span{color:#0f0;font-size:13px}
.status-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-left:8px}
.status-dot.ok{background:#0f0}
.status-dot.err{background:#f44}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:24px}
.card{background:#16213e;border:1px solid #2a2a4a;border-radius:10px;padding:20px}
.card .label{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px}
.card .value{font-size:28px;font-weight:bold;color:#0f0}
.card .unit{font-size:14px;color:#666;margin-left:4px}
table{width:100%;border-collapse:collapse;background:#16213e;border-radius:10px;overflow:hidden;border:1px solid #2a2a4a}
th{text-align:left;padding:12px 16px;background:#12122a;color:#888;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;border-bottom:1px solid #2a2a4a}
td{padding:10px 16px;border-bottom:1px solid #1a1a2e;font-size:14px;font-family:monospace}
tr:last-child td{border-bottom:none}
.section-title{font-size:16px;color:#ccc;margin:24px 0 12px;font-weight:600}
.refresh-note{font-size:12px;color:#555;margin-top:16px}
</style></head><body>
<div class="header">
  <h1>infergo</h1><span>Dashboard</span>
  <span class="status-dot ok" id="statusDot" title="Connected"></span>
</div>
<div class="grid">
  <div class="card"><div class="label">Requests Total</div><div class="value" id="reqTotal">--</div></div>
  <div class="card"><div class="label">Avg Latency</div><div class="value" id="avgLatency">--<span class="unit">ms</span></div></div>
  <div class="card"><div class="label">Active Models</div><div class="value" id="activeModels">--</div></div>
  <div class="card"><div class="label">Uptime</div><div class="value" id="uptime">--</div></div>
  <div class="card"><div class="label">Queue Depth</div><div class="value" id="queueDepth">--</div></div>
  <div class="card"><div class="label">Cache Hit Rate</div><div class="value" id="cacheRate">--<span class="unit">%</span></div></div>
</div>
<div class="section-title">Raw Metrics</div>
<table>
  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody id="metricsBody"><tr><td colspan="2">Loading...</td></tr></tbody>
</table>
<div class="refresh-note" id="refreshNote">Auto-refreshing every 2s</div>
<script>
const reqTotal=document.getElementById('reqTotal');
const avgLatency=document.getElementById('avgLatency');
const activeModels=document.getElementById('activeModels');
const uptimeEl=document.getElementById('uptime');
const queueDepth=document.getElementById('queueDepth');
const cacheRate=document.getElementById('cacheRate');
const metricsBody=document.getElementById('metricsBody');
const statusDot=document.getElementById('statusDot');
const refreshNote=document.getElementById('refreshNote');
const startTime=Date.now();

function parseMetrics(text){
  const metrics={};
  const lines=text.split('\n');
  for(const line of lines){
    if(line.startsWith('#')||!line.trim())continue;
    const parts=line.split(' ');
    if(parts.length>=2){
      metrics[parts[0]]=parseFloat(parts[1]);
    }
  }
  return metrics;
}

function fmtUptime(ms){
  const s=Math.floor(ms/1000);
  const h=Math.floor(s/3600);
  const m=Math.floor((s%3600)/60);
  const sec=s%60;
  if(h>0)return h+'h '+m+'m';
  if(m>0)return m+'m '+sec+'s';
  return sec+'s';
}

async function refresh(){
  try{
    const r=await fetch('/metrics');
    const text=await r.text();
    const m=parseMetrics(text);

    // Requests total: sum all infergo_requests_total entries
    let total=0;
    const lines=text.split('\n');
    for(const line of lines){
      if(line.startsWith('infergo_requests_total')){
        const v=parseFloat(line.split(' ').pop());
        if(!isNaN(v))total+=v;
      }
    }
    reqTotal.textContent=total>0?total.toFixed(0):'0';

    // Avg latency from histogram sum/count
    let dSum=0,dCount=0;
    for(const line of lines){
      if(line.startsWith('infergo_request_duration_seconds_sum')){
        const v=parseFloat(line.split(' ').pop());
        if(!isNaN(v))dSum+=v;
      }
      if(line.startsWith('infergo_request_duration_seconds_count')){
        const v=parseFloat(line.split(' ').pop());
        if(!isNaN(v))dCount+=v;
      }
    }
    if(dCount>0){
      avgLatency.innerHTML=(dSum/dCount*1000).toFixed(1)+'<span class="unit">ms</span>';
    }else{
      avgLatency.innerHTML='--<span class="unit">ms</span>';
    }

    // Queue depth
    const qd=m['infergo_queue_depth'];
    queueDepth.textContent=qd!==undefined?qd.toFixed(0):'0';

    // Cache hit rate
    const hits=m['infergo_cache_hits_total']||0;
    const misses=m['infergo_cache_misses_total']||0;
    if(hits+misses>0){
      cacheRate.innerHTML=((hits/(hits+misses))*100).toFixed(1)+'<span class="unit">%</span>';
    }else{
      cacheRate.innerHTML='--<span class="unit">%</span>';
    }

    // Uptime (client-side approximation from page load)
    uptimeEl.textContent=fmtUptime(Date.now()-startTime);

    // Active models from /v1/models
    try{
      const mr=await fetch('/v1/models');
      const mj=await mr.json();
      activeModels.textContent=(mj.data?mj.data.length:0).toString();
    }catch(e){activeModels.textContent='--';}

    // Raw metrics table: show important infergo_ lines
    const rows=[];
    for(const line of lines){
      if(line.startsWith('#')||!line.trim())continue;
      if(line.startsWith('infergo_')||line.startsWith('go_')||line.startsWith('process_')){
        const parts=line.split(' ');
        if(parts.length>=2){
          rows.push('<tr><td>'+parts[0].replace(/</g,'&lt;')+'</td><td>'+parts[1]+'</td></tr>');
        }
      }
    }
    metricsBody.innerHTML=rows.length>0?rows.join(''):'<tr><td colspan="2">No metrics available</td></tr>';

    statusDot.className='status-dot ok';
    statusDot.title='Connected';
    refreshNote.textContent='Auto-refreshing every 2s — last update: '+new Date().toLocaleTimeString();
  }catch(e){
    statusDot.className='status-dot err';
    statusDot.title='Error: '+e.message;
    refreshNote.textContent='Fetch failed: '+e.message;
  }
}

refresh();
setInterval(refresh,2000);
</script></body></html>`
