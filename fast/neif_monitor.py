#!/usr/bin/env python3
"""neif_monitor.py OUT_DIR [SECONDS]: every 10 s one line in OUT_DIR/monitor.tsv while the split
evaluation runs (cgroup v1 or v2): container memory, page cache, OOM kills, CPU throttling, load
average, busy share of ALL host CPUs (/proc/stat), container cpu seconds, file system, real size
of OUT_DIR, cmix processes, the pass named in OUT_DIR/.pass and the sum of the 'progress' percent
of the newest log of every split (its change per interval shows a throughput collapse)."""
import glob, os, re, subprocess, sys, time
out = sys.argv[1]; step = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
CG = "/sys/fs/cgroup/"
def rd(*names):
    for n in names:
        try: return open(CG + n).read()
        except OSError: pass
    return ""
def key(text, k):
    for ln in text.splitlines():
        p = ln.split()
        if len(p) == 2 and p[0] == k: return p[1]
    return "0"
def host():
    p = open("/proc/stat").readline().split()[1:]
    v = [int(x) for x in p]; return sum(v), v[3] + v[4]
def progress():
    tot = 0.0; n = 0
    for d in glob.glob(out + "/c*/"):
        logs = [f for f in glob.glob(d + "*log") if os.path.isfile(f)]
        if not logs: continue
        f = max(logs, key=os.path.getmtime)
        try:
            with open(f, "rb") as h:
                h.seek(max(0, os.path.getsize(f) - 400)); t = h.read().decode("latin1")
            m = re.findall(r"progress\s+([0-9.]+)", t)
            if m: tot += float(m[-1]); n += 1
        except OSError: pass
    return "%.2f" % tot, n
path = out + "/monitor.tsv"
if not os.path.exists(path):
    open(path, "w").write("utc\tpass\tcgroup_mem_bytes\tcache_bytes\toom_kills\tnr_throttled\tloadavg1\thost_busy_share\tcgroup_cpu_s\tfs_used_kb\tfs_avail_kb\tout_du_kb\tcmix_procs\tprogress_sum\tprogress_logs\n")
t0, i0 = host()
while True:
    time.sleep(step)
    try:
        t1, i1 = host(); busy = 1 - (i1 - i0) / max(1, t1 - t0); t0, i0 = t1, i1
        stat = rd("memory.stat", "memory/memory.stat")
        cpu = rd("cpuacct/cpuacct.usage"); cpus = int(cpu) / 1e9 if cpu.strip() else int(key(rd("cpu.stat"), "usage_usec")) / 1e6
        st = os.statvfs(out)
        du = subprocess.run(["du", "-sk", out], capture_output=True, text=True).stdout.split("\t")[0]
        n = subprocess.run(["pgrep", "-c", "-x", "cmix"], capture_output=True, text=True).stdout.strip()
        try: ps = open(out + "/.pass").read().strip()
        except OSError: ps = "-"
        pr, pn = progress()
        row = [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), ps, rd("memory.current", "memory/memory.usage_in_bytes").strip(),
               key(stat, "file") if key(stat, "file") != "0" else key(stat, "cache"),
               key(rd("memory.events", "memory/memory.oom_control"), "oom_kill"), key(rd("cpu.stat", "cpu/cpu.stat"), "nr_throttled"),
               open("/proc/loadavg").read().split()[0], "%.3f" % busy, "%.1f" % cpus,
               (st.f_blocks - st.f_bfree) * st.f_frsize // 1024, st.f_bavail * st.f_frsize // 1024, du, n, pr, pn]
        open(path, "a").write("\t".join(str(x) for x in row) + "\n")
    except Exception as ex:
        open(path, "a").write("# %s\n" % ex)
