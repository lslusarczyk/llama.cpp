#!/usr/bin/env python3
"""Summarize bench-openvino-phase-split-sweep CSV; find hybrid wins."""
import csv
import sys
from collections import defaultdict

def main():
    rows = list(csv.DictReader(sys.stdin))
    by_pg = defaultdict(dict)
    for r in rows:
        by_pg[r["pg"]][r["config"]] = float(r["avg_ts"])

    print("pg\tbest_single\tbest_single_cfg\tbest_hybrid\thybrid_cfg\thybrid_vs_single")
    winners = []
    for pg in sorted(by_pg.keys(), key=lambda s: (int(s.split(",")[0]), int(s.split(",")[1]))):
        d = by_pg[pg]
        singles = {k: d[k] for k in ("all_cpu", "all_igpu") if k in d}
        hybrids = {k: d[k] for k in ("split_cpu_pp_igpu_tg", "split_igpu_pp_cpu_tg") if k in d}
        if not singles or not hybrids:
            continue
        bs_cfg = max(singles, key=singles.get)
        bs = singles[bs_cfg]
        bh_cfg = max(hybrids, key=hybrids.get)
        bh = hybrids[bh_cfg]
        ratio = bh / bs if bs > 0 else 0.0
        print(f"{pg}\t{bs:.2f}\t{bs_cfg}\t{bh:.2f}\t{bh_cfg}\t{ratio:.3f}")
        if bh > bs * 1.02:
            winners.append((ratio, pg, bh_cfg, bs, bh, bs_cfg))

    print("\n--- hybrid > 2% vs best single ---", file=sys.stderr)
    for ratio, pg, hc, bs, bh, sc in sorted(winners, reverse=True):
        pp, tg = pg.split(",")
        print(f"  {pg} (pp={pp} tg={tg}): {hc} {bh:.1f} vs {sc} {bs:.1f} ({ratio:.3f}x)", file=sys.stderr)

if __name__ == "__main__":
    main()
