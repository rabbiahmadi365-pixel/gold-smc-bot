#!/usr/bin/env python3
"""Gold SMC bot: liquidity sweep -> CHoCH -> FVG limit entry.
Commands: backtest | live | test | chatid.  Standard library only."""
import os, sys, json, math, argparse, urllib.request, urllib.parse
from bisect import bisect_left
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo

CFG = dict(
    symbol="GC=F", pivot_n=3, lookback=48, sweep_to_break=12, fill_expiry=24,
    max_hold=60, rr=2.0, buf=0.30, min_gap=0.20, max_risk_atr=3.0, min_risk=1.0,
    cost=0.30, max_loss=50.0, lot_oz=100.0, warmup=60, recent_min=120,
    killzones=[("Europe/London", "07:00", "11:00"), ("America/New_York", "07:30", "11:30")],
)
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
TUNIS = ZoneInfo("Africa/Tunis")

# ---------- data ----------
def fetch(symbol, rng, itv="5m"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?range={rng}&interval={itv}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    r = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    return [(t, o, h, l, c) for t, o, h, l, c in zip(
        r["timestamp"], q["open"], q["high"], q["low"], q["close"])
        if None not in (o, h, l, c)]

def closed_only(B, now_ts):
    return [b for b in B if b[0] + 300 <= now_ts]

def atr_series(B, n=14):
    out, trs = [], []
    for i, b in enumerate(B):
        pc = B[i - 1][4] if i else b[4]
        trs.append(max(b[2] - b[3], abs(b[2] - pc), abs(b[3] - pc)))
        k = trs[-n:]
        out.append(sum(k) / len(k))
    return out

def pivots(B, n):
    hi, lo = [], []
    for j in range(n, len(B) - n):
        h, l = B[j][2], B[j][3]
        if all(h > B[k][2] for k in range(j - n, j)) and all(h >= B[k][2] for k in range(j + 1, j + n + 1)):
            hi.append((j + n, j, h))
        if all(l < B[k][3] for k in range(j - n, j)) and all(l <= B[k][3] for k in range(j + 1, j + n + 1)):
            lo.append((j + n, j, l))
    return hi, lo

def make_view(B, n):
    hi, lo = pivots(B, n)
    return dict(B=B, hi=hi, lo=lo, ch=[p[0] for p in hi], cl=[p[0] for p in lo])

def invert(B):
    return [(t, -o, -l, -h, -c) for t, o, h, l, c in B]

def in_kz(ts, cfg):
    for tzn, a, b in cfg["killzones"]:
        d = datetime.fromtimestamp(ts, ZoneInfo(tzn))
        h1, m1 = map(int, a.split(":")); h2, m2 = map(int, b.split(":"))
        if d.weekday() < 5 and time(h1, m1) <= d.time() < time(h2, m2):
            return True
    return False

# ---------- SMC rules (long; short = same on inverted prices) ----------
def find_long(V, atr, i, cfg):
    B = V["B"]
    for s in range(i - 1, max(i - cfg["sweep_to_break"], 1) - 1, -1):
        k = bisect_left(V["cl"], s) - 1            # last swing low confirmed before s
        if k < 0: continue
        _, pj, pl = V["lo"][k]
        if pj < s - cfg["lookback"]: continue
        if not (B[s][3] < pl and B[s][4] > pl): continue   # sweep: wick below, close back above
        kh = bisect_left(V["ch"], s) - 1           # last swing high confirmed before s
        if kh < 0: continue
        mss = V["hi"][kh][2]
        if any(B[m][4] > mss for m in range(s, i)): continue   # break must happen now, first time
        if not B[i][4] > mss: continue                         # CHoCH: close above that high
        if min(B[m][3] for m in range(s, i + 1)) < B[s][3]: continue   # no lower low after sweep
        gap = None
        for kk in range(i, s + 1, -1):                         # latest bullish FVG in the move
            if B[kk][3] > B[kk - 2][2] and B[kk][3] - B[kk - 2][2] >= cfg["min_gap"]:
                gap = (B[kk - 2][2], B[kk][3]); break
        if not gap: continue
        entry = (gap[0] + gap[1]) / 2
        if entry >= B[i][4]: continue
        stop = B[s][3] - max(cfg["buf"], 0.1 * atr[i])
        risk = entry - stop
        if risk < cfg["min_risk"] or risk > cfg["max_risk_atr"] * atr[i]: continue
        return dict(entry=entry, stop=stop, target=entry + cfg["rr"] * risk,
                    risk=risk, sweep=s, gap=gap, mss=mss)
    return None

def to_real(s):
    return dict(entry=-s["entry"], stop=-s["stop"], target=-s["target"], risk=s["risk"],
                sweep=s["sweep"], gap=(-s["gap"][1], -s["gap"][0]), mss=-s["mss"])

# ---------- simulation (same code for backtest and live) ----------
def simulate(B, cfg):
    n = len(B); atr = atr_series(B)
    VL = make_view(B, cfg["pivot_n"]); VS = make_view(invert(B), cfg["pivot_n"])
    trades, act = [], None

    def close(kind, px, i, t_close):
        nonlocal act
        sg = 1 if act["side"] == "LONG" else -1
        act["events"].append(dict(kind=kind, t=t_close, px=px))
        act["result"] = kind
        if kind == "CANCELLED":
            act["R"] = 0.0
        else:
            act["R"] = (sg * (px - act["entry"]) - cfg["cost"]) / act["risk"]
        act["state"] = "closed"; trades.append(act); act = None

    for i in range(cfg["warmup"], n):
        b = B[i]; tc = b[0] + 300
        if act:
            long_ = act["side"] == "LONG"
            if act["state"] == "pending":
                filled = b[3] <= act["entry"] if long_ else b[2] >= act["entry"]
                if filled:
                    act["state"] = "open"; act["fill_i"] = i
                    act["events"].append(dict(kind="FILLED", t=tc, px=act["entry"]))
                    if (b[3] <= act["stop"]) if long_ else (b[2] >= act["stop"]):
                        close("STOP", act["stop"], i, tc)
                elif i - act["sig_i"] >= cfg["fill_expiry"]:
                    close("CANCELLED", b[4], i, tc)
            else:
                hs = (b[3] <= act["stop"]) if long_ else (b[2] >= act["stop"])
                ht = (b[2] >= act["target"]) if long_ else (b[3] <= act["target"])
                if hs: close("STOP", act["stop"], i, tc)            # same bar both -> stop (conservative)
                elif ht: close("TARGET", act["target"], i, tc)
                elif i - act["fill_i"] >= cfg["max_hold"]: close("TIMEOUT", b[4], i, tc)
        if act is None and (not cfg.get("kz", True) or in_kz(tc, cfg)):
            side, s = "LONG", find_long(VL, atr, i, cfg)
            if not s:
                s2 = find_long(VS, atr, i, cfg)
                if s2: side, s = "SHORT", to_real(s2)
            if s:
                act = dict(id=f"{side}-{b[0]}", side=side, sig_i=i, state="pending",
                           events=[dict(kind="SIGNAL", t=tc, px=s["entry"])], **s)
    if act: trades.append(act)
    return trades

# ---------- backtest ----------
def report(trades, cfg):
    done = [t for t in trades if t.get("result") in ("TARGET", "STOP", "TIMEOUT")]
    canc = [t for t in trades if t.get("result") == "CANCELLED"]
    print(f"setups {len(trades)} | traded {len(done)} | never filled {len(canc)}")
    if not done:
        print("no completed trades"); return
    Rs = [t["R"] for t in done]
    w = [r for r in Rs if r > 0]; l = [r for r in Rs if r <= 0]
    pf = sum(w) / abs(sum(l)) if l and sum(l) != 0 else float("inf")
    eq = peak = dd = 0.0
    for r in Rs:
        eq += r; peak = max(peak, eq); dd = max(dd, peak - eq)
    avg = sum(Rs) / len(Rs)
    print(f"win rate {len(w)/len(done)*100:.0f}% | avg {avg:+.2f}R | total {sum(Rs):+.1f}R | PF {pf:.2f} | max drawdown {dd:.1f}R")
    for kind in ("TARGET", "STOP", "TIMEOUT"):
        print(f"  {kind}: {sum(1 for t in done if t['result']==kind)}", end="")
    print()
    for sd in ("LONG", "SHORT"):
        x = [t["R"] for t in done if t["side"] == sd]
        if x: print(f"  {sd}: {len(x)} trades, avg {sum(x)/len(x):+.2f}R")
    print(f"(costs included: {cfg['cost']}$/oz per trade, rr {cfg['rr']:g})")
    if len(done) < 30: print("VERDICT: sample too small to trust.")
    elif avg <= 0: print("VERDICT: NOT profitable after costs. Do not trade it as is.")
    else: print("VERDICT: positive on past data. Still run on demo before real money.")

# ---------- telegram ----------
def send(text):
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[telegram not configured]\n" + text + "\n"); return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://api.telegram.org/bot{tok}/sendMessage", data=data), timeout=20)
        return True
    except Exception as e:
        print("telegram send failed:", type(e).__name__); return False

def fmt_t(ts):
    return f"{datetime.fromtimestamp(ts, timezone.utc).astimezone(TUNIS):%H:%M} Tunis"

def msg(t, ev, cfg):
    k, side, when = ev["kind"], t["side"], fmt_t(ev["t"])
    lots = math.floor(cfg["max_loss"] / (t["risk"] * cfg["lot_oz"]) * 100) / 100
    if k == "SIGNAL":
        size = (f"{lots} lot (~${lots*cfg['lot_oz']*t['risk']:.0f} risk)" if lots >= 0.01
                else "too small: stop too far for your max loss")
        return (f"🟡 XAU {side} SETUP: place a LIMIT order\nSweep + CHoCH + FVG\n"
                f"Entry: {t['entry']:.2f}\nStop: {t['stop']:.2f}\nTarget: {t['target']:.2f} ({cfg['rr']:g}R)\n"
                f"Risk {t['risk']:.2f}/oz | Size {size}\n"
                f"Cancel it if not filled in {cfg['fill_expiry']*5//60}h.\n{when}\nDemo first. Not advice.")
    if k == "FILLED":
        return f"✅ XAU {side} FILLED at {t['entry']:.2f}\nStop {t['stop']:.2f} | Target {t['target']:.2f}\n{when}"
    if k == "TARGET":
        return f"🎯 XAU {side} TARGET hit at {ev['px']:.2f}\nResult ~{t['R']:+.2f}R\n{when}"
    if k == "STOP":
        return f"🛑 XAU {side} STOP hit at {ev['px']:.2f}\nResult ~{t['R']:+.2f}R\n{when}"
    if k == "TIMEOUT":
        return f"⏱ XAU {side} held {cfg['max_hold']*5//60}h: close it at market (~{ev['px']:.2f}, ~{t['R']:+.2f}R)\n{when}"
    return f"❌ XAU {side} setup cancelled: not filled in time. Remove the limit order.\n{when}"

def pending_alerts(trades, sent, now, cfg):
    out = []
    for t in trades:
        for ev in t["events"]:
            key = f"{t['id']}:{ev['kind']}"
            if key in sent or now - ev["t"] > cfg["recent_min"] * 60: continue
            out.append((ev["t"], key, msg(t, ev, cfg)))
    return sorted(out)

def load_state():
    try:
        with open(STATE) as f: return json.load(f)
    except Exception:
        return {"sent": []}

def live(cfg):
    now = datetime.now(timezone.utc).timestamp()
    B = closed_only(fetch(cfg["symbol"], "5d"), now)
    if len(B) < cfg["warmup"] + 10:
        print("not enough data"); return
    print(f"{len(B)} bars, last closed {(now-(B[-1][0]+300))/60:.0f} min ago")
    st = load_state(); sent = list(st.get("sent", []))
    for _, key, text in pending_alerts(simulate(B, cfg), set(sent), now, cfg):
        if send(text): sent.append(key)
    st["sent"] = sent[-400:]
    with open(STATE, "w") as f: json.dump(st, f)

def chatid():
    tok = os.environ.get("TELEGRAM_TOKEN")
    if not tok: print("set TELEGRAM_TOKEN first"); return
    r = json.load(urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/getUpdates", timeout=20))
    ids = {m["message"]["chat"]["id"] for m in r.get("result", []) if "message" in m}
    print("chat ids:", ids or "none yet: send any message to your bot first, then rerun")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "live", "test", "chatid"])
    ap.add_argument("--range", default="60d")
    ap.add_argument("--rr", type=float); ap.add_argument("--max-loss", type=float)
    ap.add_argument("--cost", type=float); ap.add_argument("--no-kz", action="store_true")
    a = ap.parse_args()
    cfg = dict(CFG)
    if a.rr: cfg["rr"] = a.rr
    if a.max_loss: cfg["max_loss"] = a.max_loss
    if a.cost is not None: cfg["cost"] = a.cost
    if a.no_kz: cfg["kz"] = False
    if a.cmd == "backtest":
        now = datetime.now(timezone.utc).timestamp()
        B = closed_only(fetch(cfg["symbol"], a.range), now)
        print(f"{len(B)} bars of 5m {cfg['symbol']}")
        report(simulate(B, cfg), cfg)
    elif a.cmd == "live": live(cfg)
    elif a.cmd == "test": print("sent" if send("✅ Gold SMC bot: Telegram works.") else "NOT sent")
    else: chatid()

if __name__ == "__main__":
    main()
