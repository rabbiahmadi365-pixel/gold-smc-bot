#!/usr/bin/env python3
"""GOLD bot. Trend-aligned ideas on XAU (ORB London/NY + SMC sweep/CHoCH/FVG + EMA pullback),
risk rules, honest backtest, Telegram alerts, self-tracking forward record.
Commands: backtest | live | test | chatid.  Standard library only, one file."""
import os, sys, json, math, random, argparse, urllib.request, urllib.parse, urllib.error
import time as T
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo

CFG = dict(
    symbol="GC=F", basis=0.0, risk_usd=50.0, lot_oz=100.0, cost=0.30,
    max_per_day=4, day_stop_r=-2.0, max_active=3,
    trend=True, sma_fast=20, sma_slow=50, mixed_risk=0.5,
    warmup=120, buf=0.30, min_risk=1.0, max_risk_atr=3.0,
    pivot_n=3, lookback=48, sweep_to_break=12, fill_expiry=24, smc_hold=60, smc_rr=2.0, min_gap=0.20,
    orb_minutes=15, orb_rr=1.5, orb_min_width=2.0, orb_max_risk_atr=8.0, orb_hold=36,
    orb_sessions=[("LDN", "Europe/London", "08:00", "11:00"), ("NY", "America/New_York", "08:20", "11:30")],
    pb_rr=1.5, pb_hold=36, pb_expiry=6, pb_cool=12,
    killzones=[("Europe/London", "07:00", "12:00"), ("America/New_York", "07:30", "12:30")],
    strategies="smc,orb,pb", recap_hour=19, recent_min=360,
)
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
TUNIS = timezone(timedelta(hours=1))
LONDON = ZoneInfo("Europe/London")
REG = {1: "Trend (daily): UP ⬆️ → LONG only", -1: "Trend (daily): DOWN ⬇️ → SHORT only",
       0: "Trend (daily): MIXED → both sides, half size"}

def clock():
    return datetime.now(timezone.utc).timestamp()

# ============================== data ==============================
def fetch(symbol, rng, itv="5m"):
    last = None
    for host in ("query1", "query2"):
        url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
               f"?range={rng}&interval={itv}")
        for _ in range(2):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                    "Accept": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
                q = r["indicators"]["quote"][0]
                return [(t, o, h, l, c) for t, o, h, l, c in zip(
                    r["timestamp"], q["open"], q["high"], q["low"], q["close"]) if None not in (o, h, l, c)]
            except Exception as e:
                last = e; T.sleep(1.5)
    raise last

def shift(B, d):
    return B if not d else [(t, o + d, h + d, l + d, c + d) for t, o, h, l, c in B]

def closed_only(B, now_ts):
    return [b for b in B if b[0] + 300 <= now_ts]

def invert(B):
    return [(t, -o, -l, -h, -c) for t, o, h, l, c in B]

def ema(vals, n):
    a, e, out = 2 / (n + 1), None, []
    for v in vals:
        e = v if e is None else e + a * (v - e)
        out.append(e)
    return out

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

def in_kz(ts, cfg):
    for tzn, a, b in cfg["killzones"]:
        d = datetime.fromtimestamp(ts, ZoneInfo(tzn))
        h1, m1 = map(int, a.split(":")); h2, m2 = map(int, b.split(":"))
        if d.weekday() < 5 and time(h1, m1) <= d.time() < time(h2, m2):
            return True
    return False

# ============================== daily trend regime ==============================
def make_regime(D, cfg):
    """+1 up, -1 down, 0 mixed, from daily SMA20/SMA50. Uses only daily bars that started >= 24h
    before the moment asked (a daily bar lasts < 24h), so there is no look-ahead."""
    if not cfg["trend"]:
        return lambda t: None
    ts = [d[0] for d in D]; cl = [d[4] for d in D]
    f, s = cfg["sma_fast"], cfg["sma_slow"]
    st = [0] * len(D)
    for k in range(s - 1, len(D)):
        sf = sum(cl[k + 1 - f:k + 1]) / f; ss = sum(cl[k + 1 - s:k + 1]) / s
        st[k] = -1 if (cl[k] < sf < ss) else (1 if (cl[k] > sf > ss) else 0)
    def reg(t):
        k = bisect_right(ts, t - 24 * 3600) - 1
        return st[k] if k >= 0 else 0
    return reg

def allowed(r, cfg):
    if r is None: return {"LONG", "SHORT"}, 1.0
    if r > 0: return {"LONG"}, 1.0
    if r < 0: return {"SHORT"}, 1.0
    return {"LONG", "SHORT"}, cfg["mixed_risk"]

# ============================== strategies ==============================
def find_long(V, atr, i, cfg):
    """SMC (long on the given view; short = same code on inverted prices):
    liquidity sweep of a swing low -> CHoCH (close above last swing high) -> limit at the FVG middle."""
    B = V["B"]
    for s in range(i - 1, max(i - cfg["sweep_to_break"], 1) - 1, -1):
        k = bisect_left(V["cl"], s) - 1
        if k < 0: continue
        _, pj, pl = V["lo"][k]
        if pj < s - cfg["lookback"]: continue
        if not (B[s][3] < pl and B[s][4] > pl): continue
        kh = bisect_left(V["ch"], s) - 1
        if kh < 0: continue
        mss = V["hi"][kh][2]
        if any(B[m][4] > mss for m in range(s, i)): continue
        if not B[i][4] > mss: continue
        if min(B[m][3] for m in range(s, i + 1)) < B[s][3]: continue
        gap = None
        for kk in range(i, s + 1, -1):
            if B[kk][3] > B[kk - 2][2] and B[kk][3] - B[kk - 2][2] >= cfg["min_gap"]:
                gap = (B[kk - 2][2], B[kk][3]); break
        if not gap: continue
        entry = (gap[0] + gap[1]) / 2
        if entry >= B[i][4]: continue
        stop = B[s][3] - max(cfg["buf"], 0.1 * atr[i])
        risk = entry - stop
        if risk < cfg["min_risk"] or risk > cfg["max_risk_atr"] * atr[i]: continue
        return dict(entry=entry, stop=stop, target=entry + cfg["smc_rr"] * risk, risk=risk)
    return None

def pb_long(Bv, e20, e50, atr, i, cfg):
    """EMA pullback: uptrend (EMA20>EMA50, EMA50 rising), price dipped to EMA20, bullish rejection
    candle closes back above it -> buy-stop just above that candle, stop under the last 6 bars."""
    if i < 10: return None
    b, a = Bv[i], atr[i]
    if not (e20[i] > e50[i] and e50[i] > e50[i - 6]): return None
    if not (b[4] > e20[i] and b[4] > b[1] and b[4] - b[1] >= 0.3 * a): return None
    if min(Bv[i][3], Bv[i - 1][3]) > e20[i] + 0.1 * a: return None
    entry = b[2] + 0.1
    stop = min(x[3] for x in Bv[i - 5:i + 1]) - max(cfg["buf"], 0.1 * a)
    risk = entry - stop
    if risk < cfg["min_risk"] or risk > cfg["max_risk_atr"] * a: return None
    return dict(entry=entry, stop=stop, target=entry + cfg["pb_rr"] * risk, risk=risk)

def flip(s):
    return dict(entry=-s["entry"], stop=-s["stop"], target=-s["target"], risk=s["risk"])

def orb_calendar(ts, cfg):
    cal = {}
    for name, tzn, o, e in cfg["orb_sessions"]:
        tz = ZoneInfo(tzn)
        oh, om = map(int, o.split(":")); eh, em = map(int, e.split(":"))
        for d in sorted({datetime.fromtimestamp(x, tz).date() for x in ts}):
            if d.weekday() >= 5: continue
            S = datetime.combine(d, time(oh, om), tz).timestamp()
            E = S + cfg["orb_minutes"] * 60
            X = datetime.combine(d, time(eh, em), tz).timestamp()
            cal.setdefault(int(E), []).append((name, S, E, X, d))
    return cal

def make_orb(B, atr, ts, sess, sides, mult, r, cfg):
    name, S, E, X, d = sess
    i0, i1 = bisect_left(ts, S), bisect_left(ts, E)
    if i1 - i0 != cfg["orb_minutes"] // 5 or ts[i1 - 1] != E - 300: return None
    hi = max(b[2] for b in B[i0:i1]); lo = min(b[3] for b in B[i0:i1])
    a = atr[i1 - 1]; buf = max(cfg["buf"], 0.1 * a)
    width = hi - lo; risk = width + buf
    if width < cfg["orb_min_width"] or risk > cfg["orb_max_risk_atr"] * a: return None
    rr = cfg["orb_rr"]; plan = {}
    if "LONG" in sides: plan["LONG"] = (hi, lo - buf, hi + rr * risk)
    if "SHORT" in sides: plan["SHORT"] = (lo, hi + buf, lo - rr * risk)
    return dict(id=f"ORB-{name}-{d}", strategy="ORB", session=name, kind="plan", state="pending",
                plan=plan, hi=hi, lo=lo, risk=risk, expiry=X, hold=cfg["orb_hold"], regime=r, mult=mult,
                events=[dict(kind="SIGNAL", t=int(E), px=hi)])

def mk_trade(strat, side, kind, s, tc, r, mult, hold, expiry):
    return dict(id=f"{strat}-{side}-{tc}", strategy=strat, side=side, kind=kind, state="pending",
                entry=s["entry"], stop=s["stop"], target=s["target"], risk=s["risk"],
                expiry=expiry, hold=hold, regime=r, mult=mult,
                events=[dict(kind="SIGNAL", t=tc, px=s["entry"])])

# ============================== trade lifecycle ==============================
def fin(t, kind, px, tc, cfg):
    t["events"].append(dict(kind=kind, t=tc, px=px)); t["result"] = kind; t["state"] = "closed"
    if kind == "CANCELLED":
        t["R"] = t["Rw"] = 0.0
    else:
        sg = 1 if t["side"] == "LONG" else -1
        t["R"] = (sg * (px - t["entry"]) - cfg["cost"]) / t["risk"]
        t["Rw"] = t["R"] * t["mult"]

def step(t, j, B, cfg):
    """Feed closed bar j to a trade. Returns True when the trade is finished."""
    b = B[j]; tc = b[0] + 300
    if t["state"] == "pending":
        if b[0] >= t["expiry"]:
            fin(t, "CANCELLED", b[4], int(t["expiry"]), cfg); return True
        if t["kind"] == "plan":
            up = "LONG" in t["plan"] and b[2] >= t["plan"]["LONG"][0]
            dn = "SHORT" in t["plan"] and b[3] <= t["plan"]["SHORT"][0]
            if not (up or dn): return False
            side = ("LONG" if abs(b[1] - t["plan"]["LONG"][0]) <= abs(b[1] - t["plan"]["SHORT"][0]) else "SHORT") \
                if (up and dn) else ("LONG" if up else "SHORT")
            e, s_, g = t["plan"][side]
            t.update(side=side, entry=e, stop=s_, target=g)
            filled = True
        elif t["kind"] == "limit":
            filled = (b[3] <= t["entry"]) if t["side"] == "LONG" else (b[2] >= t["entry"])
        else:
            filled = (b[2] >= t["entry"]) if t["side"] == "LONG" else (b[3] <= t["entry"])
        if not filled: return False
        t["state"] = "open"; t["fill_j"] = j
        t["events"].append(dict(kind="FILLED", t=tc, px=t["entry"]))
        if (b[3] <= t["stop"]) if t["side"] == "LONG" else (b[2] >= t["stop"]):
            fin(t, "STOP", t["stop"], tc, cfg); return True
        return False
    long_ = t["side"] == "LONG"
    if (b[3] <= t["stop"]) if long_ else (b[2] >= t["stop"]):
        fin(t, "STOP", t["stop"], tc, cfg); return True            # both in one bar -> stop (conservative)
    if (b[2] >= t["target"]) if long_ else (b[3] <= t["target"]):
        fin(t, "TARGET", t["target"], tc, cfg); return True
    if j - t["fill_j"] >= t["hold"]:
        fin(t, "TIMEOUT", b[4], tc, cfg); return True
    return False

def simulate(B, cfg, reg):
    n = len(B); atr = atr_series(B); ts = [b[0] for b in B]
    cl = [b[4] for b in B]; e20 = ema(cl, 20); e50 = ema(cl, 50)
    Bi = invert(B); e20i = [-x for x in e20]; e50i = [-x for x in e50]
    VL = make_view(B, cfg["pivot_n"]); VS = make_view(Bi, cfg["pivot_n"])
    strat = cfg["strategies"].lower()
    cal = orb_calendar(ts, cfg) if "orb" in strat else {}
    active, done, last_pb = [], [], -10 ** 9
    for i in range(cfg["warmup"], n):
        b = B[i]; tc = int(b[0] + 300)
        for t in active[:]:
            if step(t, i, B, cfg):
                active.remove(t); done.append(t)
        r = reg(tc); sides, mult = allowed(r, cfg)
        for sess in cal.get(tc, []):
            t = make_orb(B, atr, ts, sess, sides, mult, r, cfg)
            if t: active.append(t)
        if not in_kz(tc, cfg): continue
        if "smc" in strat and not any(a["strategy"] == "SMC" for a in active):
            side, s = None, None
            if "LONG" in sides:
                s = find_long(VL, atr, i, cfg); side = "LONG" if s else None
            if not s and "SHORT" in sides:
                q = find_long(VS, atr, i, cfg)
                if q: s, side = flip(q), "SHORT"
            if s:
                active.append(mk_trade("SMC", side, "limit", s, tc, r, mult, cfg["smc_hold"],
                                       tc + cfg["fill_expiry"] * 300))
        if "pb" in strat and i - last_pb >= cfg["pb_cool"] and not any(a["strategy"] == "PB" for a in active):
            side, p = None, None
            if "LONG" in sides:
                p = pb_long(B, e20, e50, atr, i, cfg); side = "LONG" if p else None
            if not p and "SHORT" in sides:
                q = pb_long(Bi, e20i, e50i, atr, i, cfg)
                if q: p, side = flip(q), "SHORT"
            if p:
                last_pb = i
                active.append(mk_trade("PB", side, "stop", p, tc, r, mult, cfg["pb_hold"],
                                       tc + cfg["pb_expiry"] * 300))
    return done + active

def closed_by(t, ts):
    return t.get("state") == "closed" and t["events"][-1]["t"] <= ts

def apply_rules(trades, cfg):
    """Daily discipline: max ideas per day, stop for the day after -2R, max simultaneous ideas."""
    kept = []
    for t in sorted(trades, key=lambda x: x["events"][0]["t"]):
        ts = t["events"][0]["t"]; day = datetime.fromtimestamp(ts, LONDON).date()
        same = [k for k in kept if k["day"] == day and not (k.get("result") == "CANCELLED" and closed_by(k, ts))]
        if len(same) >= cfg["max_per_day"]: continue
        realized = sum(k["Rw"] for k in kept if k["day"] == day and k.get("result") in ("TARGET", "STOP", "TIMEOUT")
                       and closed_by(k, ts))
        if realized <= cfg["day_stop_r"]: continue
        if sum(1 for k in kept if not closed_by(k, ts)) >= cfg["max_active"]: continue
        t["day"] = day; kept.append(t)
    return kept

# ============================== statistics / backtest ==============================
def boot_ci(vals):
    if len(vals) < 8: return None
    rnd = random.Random(1); m = len(vals)
    means = sorted(sum(vals[rnd.randrange(m)] for _ in range(m)) / m for _ in range(2000))
    return means[50], means[1949]

def summarize(trades):
    done = [t for t in trades if t.get("result") in ("TARGET", "STOP", "TIMEOUT")]
    R = [t["R"] for t in done]; Rw = [t["Rw"] for t in done]
    eq = pk = dd = 0.0
    for x in Rw:
        eq += x; pk = max(pk, eq); dd = max(dd, pk - eq)
    wins = sum(x for x in Rw if x > 0); loss = -sum(x for x in Rw if x <= 0)
    return dict(done=done, n=len(done), R=R, avg=(sum(R) / len(R) if R else 0.0), tot=sum(Rw), dd=dd,
                win=(sum(1 for x in R if x > 0) / len(R) if R else 0.0),
                pf=(wins / loss if loss > 0 else float("inf")),
                nocancel=sum(1 for t in trades if t.get("result") == "CANCELLED"))

def report(trades, label, days, brief=False):
    s = summarize(trades)
    print(f"== {label} ==")
    print(f"ideas {len(trades)} | traded {s['n']} ({s['n']/days:.1f}/day) | never filled {s['nocancel']}")
    if not s["n"]:
        print("no completed trades"); return s
    ci = boot_ci(s["R"])
    cis = f" (95% range {ci[0]:+.2f}..{ci[1]:+.2f})" if ci else ""
    print(f"avg {s['avg']:+.2f}R{cis} | win {s['win']*100:.0f}% | PF {s['pf']:.2f} | total {s['tot']:+.1f}R | maxDD {s['dd']:.1f}R")
    if not brief:
        for sd in ("LONG", "SHORT"):
            x = [t["R"] for t in s["done"] if t["side"] == sd]
            if x: print(f"  {sd}: {len(x)} trades, avg {sum(x)/len(x):+.2f}R")
        for k in ("TARGET", "STOP", "TIMEOUT"):
            print(f"  {k}: {sum(1 for t in s['done'] if t['result']==k)}", end="")
        print()
    return s

def backtest(cfg, rng):
    now = clock()
    try: B = shift(closed_only(fetch(cfg["symbol"], rng, "5m"), now), cfg["basis"])
    except Exception as e:
        print("Yahoo data failed:", type(e).__name__, "- check the symbol / internet and retry."); return
    try: D = fetch(cfg["symbol"], "2y", "1d")
    except Exception as e:
        print("daily data failed:", type(e).__name__, "-> trend filter treated as MIXED"); D = []
    days = len({datetime.fromtimestamp(b[0], LONDON).date() for b in B
                if datetime.fromtimestamp(b[0], LONDON).weekday() < 5}) or 1
    print(f"{len(B)} bars of 5m {cfg['symbol']} | {days} trading days | costs {cfg['cost']}$/oz per trade")
    c0 = dict(cfg, trend=False); base = apply_rules(simulate(B, c0, make_regime(D, c0)), c0)
    c1 = dict(cfg, trend=True); reg = make_regime(D, c1)
    sysm = apply_rules(simulate(B, c1, reg), c1)
    rs = [reg(b[0] + 300) for b in B[cfg["warmup"]::288]]
    print(f"daily regime over the period: UP {rs.count(1)} | DOWN {rs.count(-1)} | MIXED {rs.count(0)} (sampled days)")
    report(base, "BASELINE (no trend filter, same rules)", days, brief=True)
    s = report(sysm, "SYSTEM (trend-aligned + daily rules)", days)
    for st in ("ORB", "SMC", "PB"):
        report([t for t in sysm if t["strategy"] == st], f"  {st}", days, brief=True)
    mid = sorted(t["events"][0]["t"] for t in sysm)[len(sysm) // 2] if sysm else 0
    a = summarize([t for t in sysm if t["events"][0]["t"] < mid]); b = summarize([t for t in sysm if t["events"][0]["t"] >= mid])
    print(f"first half avg {a['avg']:+.2f}R ({a['n']}) | second half avg {b['avg']:+.2f}R ({b['n']})")
    ci = boot_ci(s["R"])
    if s["n"] < 30: v = "VERDICT: sample too small to trust."
    elif s["avg"] <= 0: v = "VERDICT: NOT profitable after costs. Do not trade it."
    elif b["avg"] <= 0: v = "VERDICT: positive overall but the second half is not. Unstable: demo only."
    elif ci and ci[0] <= 0: v = "VERDICT: positive, but not distinguishable from luck yet. Demo only."
    else: v = "VERDICT: positive and statistically meaningful on this period. Still demo first."
    print(v)
    if s["n"] / days < 2: print("NOTE: fewer than 2 trades/day with these filters (quality over quantity).")

# ============================== telegram + messages ==============================
def send(text):
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[telegram not configured]\n" + text + "\n"); return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://api.telegram.org/bot{tok}/sendMessage", data=data), timeout=20)
        return True
    except urllib.error.HTTPError as e:
        try: why = json.loads(e.read().decode()).get("description", "")
        except Exception: why = ""
        hint = {401: "token invalid or revoked: update the TELEGRAM_TOKEN secret", 400: "check TELEGRAM_CHAT_ID / press Start on the bot",
                403: "the bot is blocked: press Start on the bot", 404: "token malformed"}.get(e.code, "")
        print(f"telegram send failed: HTTP {e.code} {why} -> {hint}"); return False
    except Exception as e:
        print("telegram send failed:", type(e).__name__); return False

def fmt_t(ts):
    return f"{datetime.fromtimestamp(ts, timezone.utc).astimezone(TUNIS):%H:%M} Tunis"

def basis_note(cfg):
    if cfg["basis"]:
        return f"الأسعار معدّلة بـ {cfg['basis']:+.1f} باش تطابق شارت البروكر (تقريبي).\n"
    if cfg["symbol"] == "GC=F":
        return "⚠️ الأسعار من futures (GC=F): قارنها بسعر البروكر، وإلا استعمل BASIS.\n"
    return ""

def sizing(t, cfg):
    lots = math.floor(cfg["risk_usd"] * t["mult"] / (t["risk"] * cfg["lot_oz"]) * 100) / 100
    if lots >= 0.01: return f"{lots} lot (~${lots*cfg['lot_oz']*t['risk']:.0f} risk)"
    return "الـ stop بعيد على أقصى خسارتك: خلّيها"

def msg(t, ev, cfg):
    k, when = ev["kind"], fmt_t(ev["t"])
    tag = t["strategy"] + (f" {t['session']}" if t["strategy"] == "ORB" else "")
    side = t.get("side", "")
    if k == "SIGNAL":
        if "plan" in t:
            lines = [f"{'BUY' if sd=='LONG' else 'SELL'} STOP {e:.2f} | SL {s_:.2f} | TP {g:.2f}"
                     for sd, (e, s_, g) in t["plan"].items()]
            body = f"Range {t['lo']:.2f} - {t['hi']:.2f}\n" + "\n".join(lines)
            if len(lines) > 1: body += "\nأول واحد يدخل، لغي الثاني."
        else:
            order = {("limit", "LONG"): "BUY LIMIT", ("limit", "SHORT"): "SELL LIMIT",
                     ("stop", "LONG"): "BUY STOP", ("stop", "SHORT"): "SELL STOP"}[(t["kind"], t["side"])]
            body = f"{order} {t['entry']:.2f} | SL {t['stop']:.2f} | TP {t['target']:.2f}"
        return (f"🟡 XAU {tag} [{cfg['symbol']}]\n{REG.get(t['regime'], '')}\n{body}\n"
                f"Risk {t['risk']:.2f}/oz | {sizing(t, cfg)}\nصالحة لين {fmt_t(t['expiry'])}\n{when}\n"
                f"{basis_note(cfg)}Demo first. Not advice.").replace("\n\n", "\n")
    if k == "FILLED":
        extra = "\nلغي الأمر الثاني." if "plan" in t and len(t["plan"]) > 1 else ""
        return f"✅ XAU {tag} {side} FILLED at {t['entry']:.2f}\nSL {t['stop']:.2f} | TP {t['target']:.2f}{extra}\n{when}"
    if k == "TARGET": return f"🎯 XAU {tag} {side} TARGET {ev['px']:.2f}\nResult ~{t['R']:+.2f}R\n{when}"
    if k == "STOP": return f"🛑 XAU {tag} {side} STOP {ev['px']:.2f}\nResult ~{t['R']:+.2f}R\n{when}"
    if k == "TIMEOUT":
        return f"⏱ XAU {tag} {side}: سكّرها على السوق (~{ev['px']:.2f}, ~{t['R']:+.2f}R)\n{when}"
    return f"❌ XAU {tag}: ما دخلتش. لغي الأمر.\n{when}"

def pending_alerts(trades, sent, now, cfg, first_run=False):
    out = []
    for t in trades:
        for ev in t["events"]:
            key = f"{t['id']}:{ev['kind']}"
            late = int((now - ev["t"]) / 60)
            if key in sent or late > cfg["recent_min"]: continue
            if ev["kind"] == "SIGNAL" and now >= t["expiry"]: continue      # order already expired: useless
            if first_run and late > 90:
                out.append((ev["t"], key, None)); continue          # old history: mark as seen, do not send
            text = msg(t, ev, cfg)
            if late > 10: text = f"⏰ متأخر {late} دقيقة\n" + text
            out.append((ev["t"], key, text))
    return sorted(out)

# ============================== state, forward record, recap ==============================
def load_state():
    try:
        with open(STATE) as f: st = json.load(f)
    except Exception:
        st = {}
    st.setdefault("sent", []); st.setdefault("hist", []); st.setdefault("recap", "")
    return st

def save_state(st):
    st["sent"] = st["sent"][-600:]; st["hist"] = st["hist"][-500:]
    with open(STATE, "w") as f: json.dump(st, f)

def record_closed(st, trades):
    have = {h["id"] for h in st["hist"]}
    for t in trades:
        if t["events"][0]["t"] < st.get("since", 0): continue          # history before the bot started is not "forward"
        if t.get("result") in ("TARGET", "STOP", "TIMEOUT") and t["id"] not in have:
            st["hist"].append(dict(id=t["id"], s=t["strategy"], side=t["side"], R=round(t["R"], 3),
                                   Rw=round(t["Rw"], 3), day=str(t["day"])))

def forward_line(st):
    h = st["hist"]
    if not h: return "السجل الفعلي: ما فماش صفقات مسكّرة بعد."
    R = [x["R"] for x in h]; tot = sum(x["Rw"] for x in h)
    s = f"السجل الفعلي: {len(h)} صفقة | متوسط {sum(R)/len(R):+.2f}R | مجموع {tot:+.1f}R | ربح {sum(1 for x in R if x>0)*100//len(R)}%"
    if len(h) >= 20:
        l20 = h[-20:]; t20 = sum(x["Rw"] for x in l20)
        s += f"\nآخر 20: {t20:+.1f}R"
        if t20 <= -8: s += "\n⚠️ الاستراتيجية تخسر في الواقع. وقّف وارجع للـ backtest."
    return s

def maybe_recap(st, trades, now, cfg):
    d = datetime.fromtimestamp(now, timezone.utc).astimezone(TUNIS)
    today = str(d.date())
    if d.weekday() >= 5 or d.hour < cfg["recap_hour"] or st["recap"] == today: return
    mine = [t for t in trades if datetime.fromtimestamp(t["events"][0]["t"], timezone.utc).astimezone(TUNIS).date() == d.date()]
    lines = [f"📅 {today}"]
    tot = 0.0
    for t in sorted(mine, key=lambda x: x["events"][0]["t"]):
        tag = t["strategy"] + (f" {t['session']}" if t["strategy"] == "ORB" else "")
        r = t.get("result")
        if r in ("TARGET", "STOP", "TIMEOUT"):
            lines.append(f"{tag} {t['side']}: {t['R']:+.2f}R {'✅' if t['R']>0 else '❌'}"); tot += t["Rw"]
        elif r == "CANCELLED": lines.append(f"{tag}: ما دخلتش")
        elif t["state"] == "open": lines.append(f"{tag} {t['side']}: مازالت مفتوحة")
        else: lines.append(f"{tag}: في الانتظار")
    lines.append(f"اليوم: {tot:+.1f}R")
    lines.append(forward_line(st))
    if send("\n".join(lines)): st["recap"] = today

def plan_times(now, cfg):
    out = []
    for name, tzn, o, e in cfg["orb_sessions"]:
        tz = ZoneInfo(tzn); oh, om = map(int, o.split(":"))
        d = datetime.fromtimestamp(now, tz).date()
        S = datetime.combine(d, time(oh, om), tz).timestamp()
        out.append(f"{name} {fmt_t(S + cfg['orb_minutes'] * 60)}")
    return " | ".join(out)

def heartbeat(st, B, reg, now, cfg):
    d = datetime.fromtimestamp(now, LONDON)
    if d.weekday() >= 5 or not (time(7, 55) <= d.time() < time(20, 0)) or st.get("beat") == str(d.date()):
        return
    if now - (B[-1][0] + 300) > 20 * 60: return                       # data not fresh: stay quiet
    r = reg(int(now))
    text = (f"☀️ Gold bot شغّال ({d.date()})\nالسعر ~{B[-1][4]:.2f} [{cfg['symbol']}]\n"
            f"{REG.get(r, 'Trend filter: off')}\nالخطط: {plan_times(now, cfg)}")
    if send(text): st["beat"] = str(d.date())

def check_outage(st, now, cfg):
    d = datetime.fromtimestamp(now, LONDON)
    last = st.get("last_ok")
    if last is None or d.weekday() >= 5 or not (7 <= d.hour < 20): return
    if now - last > 20 * 60 and st.get("outage") != str(d.date()):
        if send(f"⚠️ ما نجمتش نجيب بيانات الذهب من {int((now-last)/60)} دقيقة (Yahoo). البوت يعاود وحدو."):
            st["outage"] = str(d.date())

# ============================== live ==============================
def live(cfg):
    st = load_state()
    if "since" not in st:                       # first run of this version: forward record starts now
        st["since"] = clock(); st["hist"] = []
    st.setdefault("last_ok", clock())
    deadline = clock() + int(os.environ.get("RUN_SECONDS", "240"))
    D, d_at = [], -1e9
    while True:
        now = clock()
        try:
            if now - d_at > 3600:
                try: D = fetch(cfg["symbol"], "2y", "1d"); d_at = now
                except Exception as e: print("daily fetch failed:", type(e).__name__)
            B = shift(closed_only(fetch(cfg["symbol"], "5d", "5m"), now), cfg["basis"])
            if len(B) > cfg["warmup"] + 10:
                reg = make_regime(D, cfg)
                trades = apply_rules(simulate(B, cfg, reg), cfg)
                heartbeat(st, B, reg, now, cfg)
                sent_now = 0
                for _, key, text in pending_alerts(trades, set(st["sent"]), now, cfg, not st.get("init")):
                    if text is None or send(text):
                        st["sent"].append(key); sent_now += text is not None
                st["init"] = True; st["last_ok"] = now
                record_closed(st, trades)
                maybe_recap(st, trades, now, cfg)
                save_state(st)
                print(f"ok: {len(B)} bars, last bar {fmt_t(B[-1][0] + 300)}, regime {reg(int(now))}, "
                      f"ideas {len(trades)}, alerts sent {sent_now}")
            else:
                print("not enough data yet:", len(B), "bars")
        except Exception as e:
            print("cycle failed:", type(e).__name__, str(e)[:120])
            check_outage(st, now, cfg)
        if clock() + 35 > deadline: break
        T.sleep(30)
    save_state(st)

def chatid():
    tok = os.environ.get("TELEGRAM_TOKEN")
    if not tok: print("set TELEGRAM_TOKEN first"); return
    r = json.load(urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/getUpdates", timeout=20))
    ids = {m["message"]["chat"]["id"] for m in r.get("result", []) if "message" in m}
    print("chat ids:", ids or "none yet: send any message to your bot first")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "live", "test", "chatid"])
    ap.add_argument("--range", default="60d"); ap.add_argument("--symbol"); ap.add_argument("--basis", type=float)
    ap.add_argument("--risk", type=float); ap.add_argument("--no-trend", action="store_true")
    ap.add_argument("--strategies"); ap.add_argument("--max-day", type=int)
    a = ap.parse_args()
    cfg = dict(CFG)
    cfg["symbol"] = a.symbol or os.environ.get("SYMBOL") or cfg["symbol"]
    cfg["basis"] = a.basis if a.basis is not None else float(os.environ.get("BASIS", "0") or 0)
    cfg["risk_usd"] = a.risk or float(os.environ.get("RISK_USD", "0") or 0) or cfg["risk_usd"]
    if a.no_trend: cfg["trend"] = False
    cfg["strategies"] = a.strategies or os.environ.get("STRATEGIES") or cfg["strategies"]
    if a.max_day: cfg["max_per_day"] = a.max_day
    if a.cmd == "backtest": backtest(cfg, a.range)
    elif a.cmd == "live": live(cfg)
    elif a.cmd == "test": print("sent" if send("✅ Gold bot: Telegram works.") else "NOT sent")
    else: chatid()

if __name__ == "__main__":
    main()
