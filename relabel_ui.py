#!/usr/bin/env python3
"""Local ear-review UI for ambiguous labels. Plays each sound + a reference
example per category; you pick the label. Decisions append to
training_data/relabel/decisions.jsonl (stop/resume anytime).
Run: venv-clap/bin/python training_data/relabel_ui.py  then open the printed URL.
"""
import os, json, mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
TD="training_data"; RL=f"{TD}/relabel"; DEC=f"{RL}/decisions.jsonl"; PORT=8777
QUEUE=json.load(open(f"{RL}/review_queue.json")); EX=json.load(open(f"{RL}/examples.json"))
AMAP=json.load(open(f"{RL}/audio_map.json")); CLASSES=sorted(EX.keys())
done=set()
if os.path.exists(DEC):
    for ln in open(DEC):
        try: done.add(json.loads(ln)["hash"])
        except Exception: pass

PAGE="""<!doctype html><html><head><meta charset=utf-8><title>PulseMap relabel</title>
<style>body{font-family:system-ui;margin:0;background:#14161c;color:#e6e8ec}
#top{padding:14px 20px;background:#1b1e26;position:sticky;top:0;border-bottom:1px solid #2a2e38}
#name{color:#8a929e;font-size:13px} #cur{font-size:20px;font-weight:600;margin:6px 0}
#hint{color:#6ea0ff;font-size:14px} button{font:inherit;color:#e6e8ec;background:#252a34;border:1px solid #333a46;border-radius:8px;padding:8px 12px;cursor:pointer}
button:hover{background:#2d3340} .big{font-size:16px;padding:12px 20px;margin:8px 8px 0 0}
#grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:20px;max-width:900px}
.cell{display:flex;gap:6px;align-items:stretch} .cell .lab{flex:1;text-align:left} .cell .ex{width:38px}
.sug{border-color:#6ea0ff;background:#22304a} #prog{float:right;color:#8a929e}
kbd{background:#333a46;border-radius:4px;padding:1px 6px;font-size:12px;color:#9aa}</style></head>
<body><div id=top><span id=prog></span><div id=name></div><div id=cur></div><div id=hint></div>
<button class=big onclick=play()>▶ Replay <kbd>space</kbd></button>
<button class=big onclick=decide('keep')>Keep current <kbd>k</kbd></button>
<button class=big onclick=decide('notdrum')>Not a drum</button>
<button class=big onclick=decide('skip')>Skip <kbd>s</kbd></button></div>
<div id=grid></div><audio id=a></audio><audio id=e></audio>
<script>
let Q=[],i=0,cur=null,EX={},CLASSES=[],DONE=%DONE%;
async function load(){Q=await (await fetch('/queue')).json();next()}
function play(){let a=document.getElementById('a');a.src='/audio?h='+cur.hash+'&t='+Date.now();a.play()}
function playex(c){let e=document.getElementById('e');e.src='/audio?h='+EX[c].hash+'&t='+Date.now();e.play()}
function next(){
 if(i>=Q.length){document.getElementById('cur').textContent='All done in this batch — thank you!';document.getElementById('grid').innerHTML='';document.getElementById('prog').textContent='Reviewed '+DONE;return}
 cur=Q[i];document.getElementById('name').textContent=cur.name||cur.hash;
 document.getElementById('cur').textContent='Currently labeled: '+cur.current;
 let sug=Object.entries(cur.neighbors).sort((a,b)=>b[1]-a[1]).map(x=>x[0]);
 document.getElementById('hint').textContent='Neighbors suggest: '+sug.slice(0,3).join(', ');
 document.getElementById('prog').textContent='Reviewed '+DONE+' · item '+(i+1)+' / '+Q.length;
 let g=document.getElementById('grid');g.innerHTML='';
 CLASSES.forEach((c,n)=>{let d=document.createElement('div');d.className='cell';
  let key=sug.indexOf(c);let k=key>=0&&key<3?' ['+(key+1)+']':'';
  d.innerHTML='<button class="lab'+(sug.slice(0,3).includes(c)?' sug':'')+'" onclick="decide(\\''+c+'\\')">'+c+k+'</button><button class=ex onclick="playex(\\''+c+'\\')">▶</button>';
  g.appendChild(d)});
 play();
}
async function decide(label){await fetch('/decide',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({hash:cur.hash,current:cur.current,label:label})});DONE++;i++;next()}
document.addEventListener('keydown',e=>{
 if(!cur)return;
 if(e.key===' '){e.preventDefault();play();}
 else if(e.key==='k'){decide('keep');}
 else if(e.key==='s'){decide('skip');}
 else if(['1','2','3'].includes(e.key)){let sug=Object.entries(cur.neighbors).sort((a,b)=>b[1]-a[1]).map(x=>x[0]);
  let c=sug[+e.key-1];if(c)decide(c);}});
(async()=>{EX=await (await fetch('/examples')).json();CLASSES=Object.keys(EX).sort();load()})();
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _send(self,code,body,ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type",ctype); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u=urlparse(self.path)
        if u.path=="/": self._send(200,PAGE.replace("%DONE%",str(len(done))).encode(),"text/html")
        elif u.path=="/queue": self._send(200,json.dumps([q for q in QUEUE if q["hash"] not in done]).encode())
        elif u.path=="/examples": self._send(200,json.dumps(EX).encode())
        elif u.path=="/audio":
            h=parse_qs(u.query).get("h",[""])[0]; p=AMAP.get(h)
            if p and os.path.exists(p):
                data=open(p,"rb").read(); ct=mimetypes.guess_type(p)[0] or "audio/wav"; self._send(200,data,ct)
            else: self._send(404,b"")
        else: self._send(404,b"")
    def do_POST(self):
        n=int(self.headers.get("Content-Length",0)); body=json.loads(self.rfile.read(n))
        with open(DEC,"a") as f: f.write(json.dumps(body)+"\n")
        done.add(body["hash"]); self._send(200,b'{"ok":1}')

print(f"Review UI: http://localhost:{PORT}   ({len(QUEUE)-len(done)} left, {len(done)} done)")
print("Decisions saved live to",DEC,"— stop with Ctrl-C, resume by rerunning.")
ThreadingHTTPServer(("127.0.0.1",PORT),H).serve_forever()
