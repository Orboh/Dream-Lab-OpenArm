"""Read-only RealSense + Dream-1 probe page. Never opens the robot or CAN.

  .venv/bin/python scripts/camera_probe_web.py        # then open http://127.0.0.1:8765

Shows the three live cameras (serials from cameras.pinned_serials in config/pipeline.yaml), and an "Ask Dream"
button that sends the current overhead frame (optionally the wrist frames) to Dream-1 with a zero joint state and
plots the returned joint targets. Dream is a policy, not a detector: it has no "found the block" output, so
check that the plot changes when you move the object. Stop other programs that hold the cameras first.
"""
import asyncio, os, threading, time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from PIL import Image
from orboh.cameras.sources import list_realsense, RealSenseSource, to_jpeg
from orboh.config import load_config
from orboh.dreamlab.policy import DreamPolicyClient, PolicyError

cfg = load_config(None)
# Dream slot -> serial, through policy.camera_map and the pinned role serials
ROLES = {slot: cfg.cameras.pinned_serials[role] for slot, role in cfg.policy.camera_map.items()
         if role and cfg.cameras.pinned_serials.get(role)}
if not ROLES:
    raise SystemExit("set cameras.pinned_serials in config/pipeline.yaml (run `orboh discover-cameras` to find them)")
latest: dict[str, bytes] = {}
lock = threading.Lock()

def pump(slot, serial):
    dev = next(d for d in list_realsense() if d.serial == serial)
    src = RealSenseSource(dev, 640, 480, 30)
    while True:
        img = src.read()
        if img is not None:
            j = to_jpeg(img, 80)
            with lock: latest[slot] = j

for slot, ser in ROLES.items():
    threading.Thread(target=pump, args=(slot, ser), daemon=True).start()

_client = None
def client():
    global _client
    if _client is None:
        _client = DreamPolicyClient(cfg.policy, os.environ["DREAMLAB_API_KEY"])
    return _client

app = FastAPI()

@app.get("/")
def index(): return HTMLResponse(PAGE)

@app.get("/info")
def info(): return {"roles": ROLES}

@app.get("/snap/{slot}.jpg")
def snap(slot: str):
    with lock: j = latest.get(slot)
    return Response(j or b"", media_type="image/jpeg")

@app.get("/stream/{slot}")
async def stream(slot: str):
    async def gen():
        while True:
            with lock: j = latest.get(slot)
            if j: yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

@app.post("/ask")
def ask(prompt: str = "Pick up the wood and place it in the box", wrists: bool = False):
    with lock: snaps = dict(latest)
    sent = {"overhead": snaps.get("overhead")}
    if wrists: sent.update(wrist_l=snaps.get("wrist_l"), wrist_r=snaps.get("wrist_r"))
    with open("/tmp/last_sent.jpg", "wb") as f: f.write(sent["overhead"])
    c = client()
    try:
        c.open(prompt)
        for _ in range(5):
            try:
                c.observe(np.zeros(16), sent); t = time.time(); tr = c.get_chunk(); break
            except PolicyError as e:
                if not e.retryable: raise
                time.sleep(e.retry_after_s or 3)
        else:
            return {"error": "model still warming up"}
    except PolicyError as e:
        return {"error": str(e)}
    p = tr.positions
    return {"hz": tr.hz, "rows": len(p), "secs": round(time.time() - t, 2), "timing": c.timing(),
            "chunk": np.round(p, 4).tolist(),
            "left_travel": float(np.abs(p[-1, :7] - p[0, :7]).sum()),
            "right_travel": float(np.abs(p[-1, 8:15] - p[0, 8:15]).sum()),
            "wrists": wrists}

PAGE = """<!doctype html><meta charset=utf-8><title>Dream probe</title>
<style>body{font:14px system-ui;margin:16px;background:#111;color:#eee}
.row{display:flex;gap:12px;flex-wrap:wrap}.c{background:#1b1b1b;padding:8px;border-radius:8px}
img{width:420px;max-width:100%;display:block;border-radius:4px}
input[type=text]{width:380px;padding:6px;background:#222;color:#eee;border:1px solid #444}
button{padding:7px 14px;cursor:pointer}pre{background:#1b1b1b;padding:8px;border-radius:6px;overflow:auto}</style>
<h2>RealSense + Dream-1 probe <small style="color:#888">(read-only, robot not connected)</small></h2>
<div class=row id=cams></div><br>
<div class=c><input id=p type=text value="Pick up the wood and place it in the box">
<label><input type=checkbox id=w> include wrist cams </label>
<button onclick=ask()>Ask Dream</button> <span id=s></span></div><br>
<div class=row><div class=c><b>Frame sent</b><img id=sent></div>
<div class=c><b>Predicted joint targets (rad) over the 2.4 s chunk</b><br><canvas id=cv width=640 height=300></canvas>
<div id=leg></div></div></div><pre id=o></pre>
<script>
fetch('/info').then(r=>r.json()).then(i=>{for(const s in i.roles){cams.innerHTML+=
`<div class=c><b>${s}</b> <small>${i.roles[s]}</small><img src=/stream/${s}></div>`}});
async function ask(){s.textContent='asking Dream (a few seconds)...';
 const r=await (await fetch(`/ask?prompt=${encodeURIComponent(p.value)}&wrists=${w.checked}`,{method:'POST'})).json();
 sent.src='/snap/overhead.jpg?'+Date.now();
 if(r.error){s.textContent='';o.textContent=r.error;return}
 s.textContent=`round trip ${r.secs}s`;draw(r.chunk);
 o.textContent=`rows ${r.rows} @ ${r.hz} Hz\\nleft arm total joint travel  ${r.left_travel.toFixed(3)} rad\\nright arm total joint travel ${r.right_travel.toFixed(3)} rad\\n`+JSON.stringify(r.timing)}
function draw(ch){const g=cv.getContext('2d');g.clearRect(0,0,640,300);
 const col=i=>`hsl(${(i%8)*45},70%,${i<8?65:45}%)`;let leg='';
 for(let j=0;j<16;j++){g.strokeStyle=col(j);g.setLineDash(j<8?[]:[5,3]);g.beginPath();
  ch.forEach((row,t)=>{const x=t/(ch.length-1)*640,y=150-row[j]*100;t?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();
  leg+=`<span style="color:${col(j)}">${j<8?'L':'R'}${j%8<7?j%8+1:'grip'}</span> `}
 leg+='<br><small>solid = left arm, dashed = right arm</small>';document.getElementById('leg').innerHTML=leg}
</script>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
