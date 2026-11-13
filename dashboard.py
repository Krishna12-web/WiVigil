"""Browser dashboard for ESP32 CSI sensing.

  python dashboard_web.py --port COM6      (real boards, receiver port)
  python dashboard_web.py --demo           (simulated signal, to try the interface)

Then open http://127.0.0.1:8765 (it opens automatically).

What it shows: motion / presence (motion based), WiFi signal numbers (RSSI, CSI variation,
packet rate), a CSI heatmap, an EXPERIMENTAL breathing-rate estimate, and the class predicted by
model.joblib if it exists (made by train.py).
What it does NOT do: heart rate and body pose. Those are not possible with two ESP32 boards.
"""
import argparse
import json
import os
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from csi_common import N_SUB, WINDOW, clean, parse_csi_line, window_features

BUF = 8000            # packets kept in memory
CAL_SECONDS = 10      # empty-room calibration length
HOLD_SECONDS = 20     # how long "recent motion" is shown after motion stops
QUIET_NEEDED = 20     # seconds of stillness needed before a breathing estimate
RESP_WINDOW = 30      # seconds of data used for the breathing estimate
RESP_FS = 10.0        # resampling rate for the breathing analysis (Hz)
RESP_MIN_QUALITY = 8.0


class Engine:
    def __init__(self, model, demo):
        self.lock = threading.Lock()
        self.t, self.amp, self.rssi, self.motion = (deque(maxlen=BUF) for _ in range(4))
        self.prev = None
        self.model = model
        self.demo = demo
        self.baseline = None
        self.calib = None
        self.last_motion = 0.0
        self.total = 0
        self.state = {"ready": False, "demo": demo}
        self._resp = {"status": "calibrate", "bpm": None, "quality": 0.0, "wait": 0}
        self._resp_time = 0.0

    # ----- data in -----
    def add(self, t, rssi, amp):
        cur = amp / (float(amp.mean()) + 1e-6)
        with self.lock:
            m = 0.0 if self.prev is None else float(np.abs(cur - self.prev).mean())
            self.prev = cur
            self.t.append(t)
            self.amp.append(amp)
            self.rssi.append(rssi)
            self.motion.append(m)
            self.total += 1

    def start_calibration(self):
        with self.lock:
            self.calib = {"start": time.time(), "vals": []}

    # ----- analysis -----
    def analyze(self):
        now = time.time()
        with self.lock:
            n = len(self.t)
            if n < 30:
                self.state = {"ready": False, "demo": self.demo, "packets": self.total}
                return
            t = np.fromiter(self.t, float, n)
            motion = np.fromiter(self.motion, float, n)
            rssi = np.fromiter(self.rssi, float, n)
            amp = np.vstack(self.amp)
            total = self.total

        stale = now - t[-1] > 3.0
        recent = t > now - 1.0
        if recent.sum() < 5:
            recent = np.zeros(n, bool)
            recent[-5:] = True
        motion_now = float(motion[recent].mean())
        last2 = t > now - 2.0
        if last2.sum() < 5:
            last2 = recent
        a2 = amp[last2]
        valid = a2.mean(0) > 1e-3
        cv = float((a2[:, valid].std(0) / a2[:, valid].mean(0)).mean()) if valid.any() else 0.0
        rate = float(last2.sum() / 2.0) if (t[-1] - t[last2][0]) > 1.0 else 0.0
        rssi_now = float(rssi[last2].mean())

        # calibration
        calibrating = False
        cal_progress = 0.0
        if self.calib is not None:
            self.calib["vals"].append(motion_now)
            cal_progress = min((now - self.calib["start"]) / CAL_SECONDS, 1.0)
            calibrating = True
            if cal_progress >= 1.0:
                v = np.array(self.calib["vals"])
                thr = max(float(np.percentile(v, 95)) * 1.3, float(np.median(v)) * 1.5, 1e-6)
                self.baseline = {"median": float(np.median(v)), "thr": thr}
                self.calib = None
                self.last_motion = 0.0
                calibrating = False

        # motion / presence state
        thr = self.baseline["thr"] if self.baseline else None
        ratio = motion_now / thr if thr else None
        if calibrating:
            mstate = "CALIBRATING"
        elif thr is None:
            mstate = "UNCALIBRATED"
        elif ratio >= 1.0:
            mstate = "MOTION"
            self.last_motion = now
        elif now - self.last_motion < HOLD_SECONDS:
            mstate = "RECENT"
        else:
            mstate = "NONE"

        # breathing estimate (refreshed every 2 s)
        if now - self._resp_time >= 2.0:
            self._resp = self._respiration(t, amp, now, mstate)
            self._resp_time = now

        # model prediction
        label = None
        if self.model is not None and n >= WINDOW:
            try:
                feats = window_features(clean(amp[-WINDOW:]))
                label = str(self.model.predict(feats.reshape(1, -1))[0])
            except Exception:
                label = None

        # series for plots
        k = 10
        ms = np.convolve(motion[-600:], np.ones(k) / k, mode="valid")[-300:]
        rs = rssi[-300::3] if n >= 300 else rssi[::3]

        # heatmap: per-packet normalised amplitudes, valid subcarriers only
        tail = amp[-240:]
        tail = tail / (tail.mean(1, keepdims=True) + 1e-6)
        vcols = amp[-240:].mean(0) > 1e-3
        data = tail[::2][:, vcols]
        lo, hi = np.percentile(data, [2, 98])
        q = np.clip((data - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8).T
        heat = {"w": int(q.shape[1]), "h": int(q.shape[0]), "data": q.flatten().tolist()}

        self.state = {
            "ready": True, "demo": self.demo, "stale": bool(stale), "packets": total,
            "rate": round(rate, 1), "rssi": round(rssi_now, 1), "cv": round(cv, 4),
            "subcarriers": int(valid.sum()),
            "motion_now": round(motion_now, 5), "thr": None if thr is None else round(thr, 5),
            "ratio": None if ratio is None else round(ratio, 2), "motion_state": mstate,
            "cal_progress": round(cal_progress, 2), "resp": self._resp, "model": label,
            "series": {"motion": [round(float(x), 5) for x in ms], "rssi": [round(float(x), 1) for x in rs]},
            "heat": heat,
        }

    def _respiration(self, t, amp, now, mstate):
        if self.baseline is None:
            return {"status": "calibrate", "bpm": None, "quality": 0.0, "wait": 0}
        quiet_len = now - max(self.last_motion, float(t[0]))
        if mstate == "MOTION" or quiet_len < QUIET_NEEDED:
            return {"status": "settling", "bpm": None, "quality": 0.0,
                    "wait": int(max(QUIET_NEEDED - quiet_len, 0)) if mstate != "MOTION" else QUIET_NEEDED}
        start = max(now - RESP_WINDOW, self.last_motion + 2.0)
        idx = t >= start
        tt, A = t[idx], amp[idx]
        if len(tt) < 50 or tt[-1] - tt[0] < QUIET_NEEDED - 2:
            return {"status": "settling", "bpm": None, "quality": 0.0, "wait": 2}
        A = A[:, A.mean(0) > 1e-3]
        grid = np.arange(tt[0], tt[-1], 1.0 / RESP_FS)
        if len(grid) < 64 or A.shape[1] == 0:
            return {"status": "low", "bpm": None, "quality": 0.0, "wait": 0}
        X = np.empty((len(grid), A.shape[1]))
        for j in range(A.shape[1]):
            X[:, j] = np.interp(grid, tt, A[:, j])
        x = np.arange(len(grid))
        X = X - X.mean(0)
        coef = np.polyfit(x, X, 1)
        X = X - (np.outer(x, coef[0]) + coef[1])
        X = X * np.hanning(len(grid))[:, None]
        P = (np.abs(np.fft.rfft(X, axis=0)) ** 2).mean(axis=1)
        f = np.fft.rfftfreq(len(grid), 1.0 / RESP_FS)
        band = (f >= 0.1) & (f <= 0.5)         # peaks below about 10 per minute are rejected as drift (see below)
        pb, fb = P[band], f[band]
        if len(pb) < 4:
            return {"status": "low", "bpm": None, "quality": 0.0, "wait": 0}
        k = int(np.argmax(pb))
        quality = float(pb[k] / (np.median(pb) + 1e-12))
        # slow gain drift gives a spectrum that falls with frequency, so peaks at the band edge are rejected
        if k < 2 or k > len(pb) - 2:
            return {"status": "low", "bpm": None, "quality": round(quality, 1), "wait": 0}
        if quality >= RESP_MIN_QUALITY:
            return {"status": "ok", "bpm": round(float(fb[k] * 60), 1), "quality": round(quality, 1), "wait": 0}
        return {"status": "low", "bpm": None, "quality": round(quality, 1), "wait": 0}


# ---------------- data sources ----------------
def run_serial(engine, port, baud):
    import serial
    ser = serial.Serial(port, baud, timeout=0.2)
    while True:
        line = ser.readline().decode("utf-8", errors="ignore")
        p = parse_csi_line(line)
        if p is not None:
            engine.add(time.time(), p[0], p[1])


def run_demo(engine):
    """Simulated packets. 0-14 s empty, 14-28 s walking, 28-70 s still person breathing at 15/min."""
    rng = np.random.default_rng(1)
    base = rng.uniform(20, 40, N_SUB)
    base[:4] = 0
    t0, n = time.time(), 0
    while True:
        now = time.time()
        ph = (now - t0) % 70
        x = base + rng.normal(0, 0.4, N_SUB)
        if 14 <= ph < 28:
            x = x + 6 * np.sin(n / 2.5 + np.arange(N_SUB) / 4)
        elif ph >= 28:
            x = x + 1.2 * np.sin(2 * np.pi * 0.25 * now + np.arange(N_SUB) / 10)
        x[:4] = 0
        engine.add(now, -50 + rng.normal(0, 1), x.astype(np.float32))
        n += 1
        time.sleep(0.01)


def analysis_loop(engine):
    while True:
        try:
            engine.analyze()
        except Exception as e:  # keep the server alive
            engine.state = {"ready": False, "demo": engine.demo, "error": str(e)}
        time.sleep(0.4)


# ---------------- web page ----------------
HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wi-Fi CSI Sensing Monitor</title>
<style>
:root{--bg:#0a0f14;--panel:#0f1720;--line:#1d2a36;--text:#d6e2ee;--mute:#7f93a6;--accent:#37b6a8;--warn:#e0a63a;--bad:#d9534f;--ok:#4fb86a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Segoe UI",system-ui,Arial,sans-serif}
header{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
header h1{margin:0;font-size:15px;font-weight:600;letter-spacing:.02em}
#conn{font-size:12px;color:var(--mute)}
#demo{display:none;margin:12px 12px 0;padding:8px 12px;border:1px solid var(--bad);color:var(--bad);font-size:13px}
main{display:grid;grid-template-columns:290px 1fr 250px;gap:12px;padding:12px}
@media(max-width:1000px){main{grid-template-columns:1fr}}
.col{display:flex;flex-direction:column;gap:12px;min-width:0}
.panel{background:var(--panel);border:1px solid var(--line);padding:12px}
h2{margin:0 0 8px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);font-weight:600}
.big{font-size:26px;font-weight:600;line-height:1.2}
.sub{color:var(--mute);font-size:12px;margin-top:4px}
.row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}
.row b{font-weight:600}
.bar{height:6px;background:var(--line);margin-top:8px}
.bar>i{display:block;height:100%;width:0;background:var(--accent)}
.btn{margin-top:10px;background:transparent;border:1px solid var(--accent);color:var(--accent);padding:7px 12px;border-radius:2px;cursor:pointer;font:inherit}
.btn:disabled{opacity:.5;cursor:default}
canvas{display:block;width:100%;background:#070b10;border:1px solid var(--line)}
#heat{height:300px}#mplot{height:150px}#rplot{height:90px}
.note{color:var(--mute);font-size:11.5px;margin-top:8px}
</style></head><body>
<header><h1>Wi-Fi CSI Sensing Monitor</h1><div id="conn">connecting...</div></header>
<div id="demo">SIMULATED DATA. This is a test signal made by the program, not a measurement from the ESP32 boards.</div>
<main>
<div class="col">
  <div class="panel"><h2>Motion / presence</h2>
    <div class="big" id="mstate">-</div>
    <div class="sub" id="mindex">&nbsp;</div>
    <div class="bar"><i id="mbar"></i></div>
    <button class="btn" id="cal">Calibrate empty room (10 s)</button>
    <div class="note">Detects movement. A person sitting completely still is not detected as presence.</div>
  </div>
  <div class="panel"><h2>Respiration (experimental)</h2>
    <div class="big"><span id="bpm">--</span> <span style="font-size:14px;color:var(--mute)">breaths/min</span></div>
    <div class="sub" id="rtext">&nbsp;</div>
    <div class="bar"><i id="qbar"></i></div>
    <div class="note">Heuristic estimate from CSI amplitude. Range about 10 to 30 breaths per minute. It needs a still person close to the line between the boards. Not a medical measurement.</div>
  </div>
  <div class="panel"><h2>Model prediction</h2>
    <div class="big" id="label">-</div>
    <div class="note">From model.joblib, trained on a small dataset. Accuracy on new sessions is low.</div>
  </div>
</div>
<div class="col">
  <div class="panel"><h2>CSI amplitude (subcarrier vs time)</h2><canvas id="heat" height="260"></canvas></div>
  <div class="panel"><h2>Motion energy</h2><canvas id="mplot" height="150"></canvas>
    <div class="note">Dashed line is the detection threshold set by the calibration.</div></div>
</div>
<div class="col">
  <div class="panel"><h2>Wi-Fi signal</h2>
    <div class="row"><span>RSSI</span><b id="rssi">-</b></div>
    <div class="row"><span>CSI variation</span><b id="cv">-</b></div>
    <div class="row"><span>Motion index</span><b id="mi">-</b></div>
    <div class="row"><span>Packets per second</span><b id="rate">-</b></div>
    <div class="row"><span>Active subcarriers</span><b id="subs">-</b></div>
  </div>
  <div class="panel"><h2>RSSI history</h2><canvas id="rplot" height="90"></canvas></div>
  <div class="panel"><h2>Not available</h2>
    <div class="note">Heart rate and body pose cannot be measured with two ESP32 boards, so they are not shown.</div></div>
</div>
</main>
<script>
const $ = id => document.getElementById(id);
function fit(c){const r=window.devicePixelRatio||1;const w=Math.round(c.clientWidth*r),h=Math.round(c.clientHeight*r);if(c.width!==w||c.height!==h){c.width=w;c.height=h;}return c.getContext('2d');}
function color(v){ // 0..255 -> navy, teal, warm yellow
  const t=v/255, a=[10,20,40], b=[55,182,168], c=[240,220,120];
  let p,q,u; if(t<0.6){p=a;q=b;u=t/0.6;}else{p=b;q=c;u=(t-0.6)/0.4;}
  return [p[0]+(q[0]-p[0])*u,p[1]+(q[1]-p[1])*u,p[2]+(q[2]-p[2])*u];
}
const off=document.createElement('canvas');
function drawHeat(h){
  const c=$('heat'),ctx=fit(c); ctx.fillStyle='#070b10'; ctx.fillRect(0,0,c.width,c.height);
  if(!h||!h.w||!h.h) return;
  off.width=h.w; off.height=h.h; const octx=off.getContext('2d'); const img=octx.createImageData(h.w,h.h);
  for(let y=0;y<h.h;y++){for(let x=0;x<h.w;x++){
    const col=color(h.data[y*h.w+x]); const o=((h.h-1-y)*h.w+x)*4;
    img.data[o]=col[0];img.data[o+1]=col[1];img.data[o+2]=col[2];img.data[o+3]=255;}}
  octx.putImageData(img,0,0); ctx.imageSmoothingEnabled=false; ctx.drawImage(off,0,0,c.width,c.height);
}
function line(id,vals,thr,fixedMin){
  const c=$(id),ctx=fit(c),W=c.width,H=c.height; ctx.fillStyle='#070b10'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#1d2a36'; ctx.lineWidth=1; for(let i=1;i<4;i++){const y=H*i/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  if(!vals||vals.length<2) return;
  let lo=fixedMin!==undefined?fixedMin:0, hi=Math.max(...vals,thr?thr*1.5:0);
  if(fixedMin!==undefined){lo=Math.min(...vals)-2;hi=Math.max(...vals)+2;}
  if(hi-lo<1e-9) hi=lo+1;
  const X=i=>i/(vals.length-1)*W, Y=v=>H-4-(v-lo)/(hi-lo)*(H-8);
  ctx.strokeStyle='#37b6a8'; ctx.lineWidth=1.5*(window.devicePixelRatio||1); ctx.beginPath();
  vals.forEach((v,i)=>{i?ctx.lineTo(X(i),Y(v)):ctx.moveTo(X(i),Y(v));}); ctx.stroke();
  if(thr){ctx.strokeStyle='#e0a63a';ctx.setLineDash([6,5]);ctx.beginPath();ctx.moveTo(0,Y(thr));ctx.lineTo(W,Y(thr));ctx.stroke();ctx.setLineDash([]);}
}
const MS={MOTION:['Motion detected','#37b6a8'],RECENT:['Motion in last 20 s','#e0a63a'],NONE:['No motion','#7f93a6'],UNCALIBRATED:['Not calibrated','#e0a63a'],CALIBRATING:['Calibrating...','#e0a63a']};
function render(s){
  $('demo').style.display=s.demo?'block':'none';
  if(!s.ready){$('conn').textContent=s.error?('error: '+s.error):'waiting for CSI packets...';return;}
  $('conn').textContent=s.stale?'no packets in the last 3 s':('receiving, '+s.packets+' packets');
  const m=MS[s.motion_state]||['-','#d6e2ee']; $('mstate').textContent=m[0]; $('mstate').style.color=m[1];
  if(s.ratio!==null){$('mindex').textContent='Motion index '+s.ratio.toFixed(2)+' (1.00 = threshold)';$('mbar').style.width=Math.min(s.ratio/2,1)*100+'%';$('mi').textContent=s.ratio.toFixed(2);}
  else{$('mindex').textContent=s.motion_state==='CALIBRATING'?'Keep the room empty and still':'Press the button with the room empty';$('mbar').style.width=(s.cal_progress*100)+'%';$('mi').textContent='-';}
  $('cal').disabled=s.motion_state==='CALIBRATING';
  const r=s.resp; $('bpm').textContent=r.bpm!==null?r.bpm.toFixed(0):'--';
  $('qbar').style.width=Math.min(r.quality/16,1)*100+'%';
  const T={calibrate:'Calibrate the empty room first.',settling:'Waiting for a still period'+(r.wait?(' ('+r.wait+' s)'):'')+'.',low:'No reliable breathing signal.',ok:'Estimate from CSI amplitude, quality '+r.quality+'.'};
  $('rtext').textContent=T[r.status]||'';
  $('label').textContent=s.model?s.model.toUpperCase():'No model loaded';
  $('rssi').textContent=s.rssi+' dBm'; $('cv').textContent=s.cv.toFixed(3); $('rate').textContent=s.rate; $('subs').textContent=s.subcarriers;
  drawHeat(s.heat); line('mplot',s.series.motion,s.thr); line('rplot',s.series.rssi,null,0);
}
async function tick(){
  try{const r=await fetch('/api/state',{cache:'no-store'});render(await r.json());}
  catch(e){$('conn').textContent='dashboard server not reachable';}
  setTimeout(tick,350);
}
$('cal').onclick=async()=>{$('cal').disabled=true;try{await fetch('/api/calibrate',{method:'POST'});}catch(e){}};
tick();
</script></body></html>
"""


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, json.dumps(engine.state).encode("utf-8"), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/api/calibrate":
                engine.start_calibration()
                self._send(200, b'{"ok":true}', "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="receiver serial port, e.g. COM6")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--demo", action="store_true", help="simulated signal, no hardware")
    ap.add_argument("--http-port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.demo and not a.port:
        ap.error("give --port COMx, or use --demo")

    model = None
    if os.path.exists("model.joblib"):
        import joblib
        model = joblib.load("model.joblib")

    engine = Engine(model, a.demo)
    if a.demo:
        threading.Thread(target=run_demo, args=(engine,), daemon=True).start()
        threading.Timer(2.0, engine.start_calibration).start()
    else:
        threading.Thread(target=run_serial, args=(engine, a.port, a.baud), daemon=True).start()
    threading.Thread(target=analysis_loop, args=(engine,), daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(engine))
    url = f"http://127.0.0.1:{a.http_port}"
    print("Dashboard running at", url, "(Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()