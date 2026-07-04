"""Read real-clip durations (audio headers only) off the volume for the data-summary table."""
import modal
VOL = "/data"
app = modal.App("doppelganger-datastats")
vol = modal.Volume.from_name("soundmatch-sr-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("soundfile", "numpy")

@app.function(image=image, volumes={VOL: vol}, cpu=16.0, timeout=30*60)
def durations():
    import csv, json, soundfile as sf
    from concurrent.futures import ThreadPoolExecutor
    from collections import Counter
    import statistics as st
    rows = list(csv.DictReader(open(f"{VOL}/manifest_ucs_paired.csv")))
    real = [r for r in rows if r["domain"] == "real"]
    def dur(r):
        try:
            i = sf.info(f"{VOL}/{r['path']}"); return (r["event"], i.frames / i.samplerate)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=32) as ex:
        got = [x for x in ex.map(dur, real) if x]
    durs = sorted(d for _, d in got)
    by_ev = {}
    for e, d in got:
        by_ev.setdefault(e, []).append(d)
    def pct(a, p): a = sorted(a); return a[min(int(p*len(a)), len(a)-1)]
    bins = [(0,1),(1,3),(3,5),(5,10),(10,30),(30,1e9)]; lab=["<1","1-3","3-5","5-10","10-30","30+"]
    hist = Counter()
    for d in durs:
        for (lo,hi),l in zip(bins,lab):
            if lo<=d<hi: hist[l]+=1; break
    out = {"n": len(durs),
           "overall": {"min":durs[0],"p10":pct(durs,.1),"median":pct(durs,.5),
                       "mean":sum(durs)/len(durs),"p90":pct(durs,.9),"max":durs[-1]},
           "hist": {l:hist[l] for l in lab},
           "per_event_median": {e: round(st.median(v),2) for e,v in by_ev.items()}}
    print(json.dumps(out))
    return out

@app.local_entrypoint()
def main():
    import json
    r = durations.remote()
    open("/home/elliott/data/doppelganger/durations.json","w").write(json.dumps(r, indent=1))
    print("saved durations.json")
