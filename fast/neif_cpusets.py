#!/usr/bin/env python3
"""neif_cpusets.py N W: N cpu sets of W logical CPUs each, for N jobs that run at the same time
with W busy processes or threads each (cmix-neif test runs; used by neif_split_eval.sh and
neif_scale.sh).

Takes the CPUs this process may use (sched_getaffinity: inside a container that is the cgroup's
cpuset) and their core / package numbers from /sys/devices/system/cpu/cpu*/topology, and prints
one line per job: a comma list for taskset -c.  The last line on stderr says what the sets are:
  distinct physical cores   every busy process has a core of its own
  SMT siblings shared       fewer cores than N x W: both hardware threads of cores are used
  oversubscribed            fewer logical CPUs than N x W: sets overlap
Option --info prints the topology summary only.
"""
import os, sys


def topo():
    allowed = sorted(os.sched_getaffinity(0))
    cores = {}
    for c in allowed:
        base = '/sys/devices/system/cpu/cpu%d/topology/' % c
        try:
            key = (int(open(base + 'physical_package_id').read()), int(open(base + 'core_id').read()))
        except OSError:
            key = (0, c)
        cores.setdefault(key, []).append(c)
    return allowed, [cores[k] for k in sorted(cores)]


def main():
    allowed, cores = topo()
    if len(sys.argv) > 1 and sys.argv[1] == '--info':
        print('logical CPUs usable: %d, physical cores: %d, threads per core: %s' % (
            len(allowed), len(cores), sorted(set(len(c) for c in cores))))
        return
    n, w = int(sys.argv[1]), int(sys.argv[2])
    need = n * w
    if len(cores) >= need:
        order = [c[0] for c in cores]
        note = 'distinct physical cores'
    else:
        # core after core, both hardware threads: a job keeps whole cores as far as possible and
        # shares at most one core with its neighbour
        order = [x for c in cores for x in c]
        note = 'SMT siblings shared' if len(order) >= need else 'oversubscribed'
    for j in range(n):
        print(','.join(str(order[(j * w + i) % len(order)]) for i in range(w)))
    sys.stderr.write('cpu sets: %d x %d on %d cores / %d logical CPUs: %s\n' % (n, w, len(cores), len(allowed), note))


main()
