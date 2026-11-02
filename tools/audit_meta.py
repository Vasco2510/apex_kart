"""F2.2: auditoria de TODO el dataset usando solo meta.json (sin videos).
Uso: python tools/audit_meta.py data/meta_all --out results/audit_dataset.md --csv results/episodios.csv
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(root):
    rows = []
    root = Path(root)
    for p in sorted(root.rglob("meta.json")):
        ready = (p.parent / "READY.json").exists()
        rel = p.parent.relative_to(root)  # grupo/[subcarpeta/]episodio: algunos grupos anidan un nivel mas
        m = json.loads(p.read_text(encoding="utf-8"))
        s, r = m["settings"], m.get("result") or {}
        rows.append({
            "group": rel.parts[0], "episode": rel.as_posix(), "ready": ready,
            "track": s["track"], "difficulty": s["difficulty"], "laps": s.get("laps"), "num_karts": s.get("num_karts"),
            "reverse": s.get("reverse"), "seconds": m.get("simulated_seconds") or m.get("num_transitions", 0) / s.get("tick_hz", 60), "device": m.get("input_device"),
            "status": r.get("status"), "finished": bool(r.get("finished")),
            "position": r.get("native_position"), "finish_time": r.get("finish_time"),
            "human_slot": m.get("human_slot"), "kart": s["kart"],
        })
    return rows


def main():
    a = argparse.ArgumentParser()
    a.add_argument("root")
    a.add_argument("--out", required=True)
    a.add_argument("--csv")
    a = a.parse_args()
    rows = load(a.root)
    ok = [r for r in rows if r["ready"]]
    hours = sum(r["seconds"] for r in ok) / 3600
    L = ["# Auditoria del dataset completo (solo metadatos)", "",
         f"- Episodios con READY.json: **{len(ok)}** de {len(rows)}",
         f"- Horas simuladas: **{hours:.1f} h**; duracion media {sum(r['seconds'] for r in ok) / max(1, len(ok)):.0f} s",
         f"- Terminaron la carrera: {sum(r['finished'] for r in ok)} ({100 * sum(r['finished'] for r in ok) / max(1, len(ok)):.0f} %)",
         f"- Posicion final media (solo terminadas): "
         f"{sum(r['position'] for r in ok if r['finished'] and r['position']) / max(1, sum(1 for r in ok if r['finished'] and r['position'])):.2f}", ""]
    for title, key in (("Dificultad", "difficulty"), ("Vueltas", "laps"), ("Karts en carrera", "num_karts"),
                       ("En reversa", "reverse"), ("Dispositivo", "device"), ("Estado", "status")):
        L.append(f"- {title}: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r[key] for r in ok).items(), key=lambda x: str(x[0]))))
    by_track = defaultdict(list)
    for r in ok:
        by_track[r["track"]].append(r)
    L += ["", "## Por pista", "", "| pista | episodios | horas | % terminadas | pos. media (terminadas) |", "|---|---|---|---|---|"]
    for t, rs in sorted(by_track.items(), key=lambda x: -sum(r["seconds"] for r in x[1])):
        fin = [r for r in rs if r["finished"] and r["position"]]
        L.append(f"| {t} | {len(rs)} | {sum(r['seconds'] for r in rs) / 3600:.2f} | {100 * sum(r['finished'] for r in rs) / len(rs):.0f} | "
                 f"{sum(r['position'] for r in fin) / len(fin):.1f} |" if fin else
                 f"| {t} | {len(rs)} | {sum(r['seconds'] for r in rs) / 3600:.2f} | {100 * sum(r['finished'] for r in rs) / len(rs):.0f} | - |")
    unseen = sorted(set(["abyss", "black_forest", "candela_city", "cocoa_temple", "cornfield_crossing", "fortmagma",
                         "gran_paradiso_island", "hacienda", "lighthouse", "mines", "minigolf", "olivermath",
                         "ravenbridge_mansion", "sandtrack", "scotland", "snowmountain", "snowtuxpeak", "stk_enterprise",
                         "volcano_island", "xr591", "zengarden"]) - set(by_track))
    L += ["", f"Pistas sin ningun episodio: {', '.join(unseen) if unseen else 'ninguna'}"]
    by_group = defaultdict(list)
    for r in ok:
        by_group[r["group"]].append(r)
    L += ["", "## Por grupo", "", "| grupo | episodios | horas | % terminadas | pos. media (terminadas) |", "|---|---|---|---|---|"]
    for g, rs in sorted(by_group.items(), key=lambda x: -sum(r["seconds"] for r in x[1])):
        fin = [r for r in rs if r["finished"] and r["position"]]
        pm = f"{sum(r['position'] for r in fin) / len(fin):.1f}" if fin else "-"
        L.append(f"| {g} | {len(rs)} | {sum(r['seconds'] for r in rs) / 3600:.2f} | {100 * sum(r['finished'] for r in rs) / len(rs):.0f} | {pm} |")
    good = [r for r in ok if r["finished"] and r["position"] and r["position"] <= 3]
    L += ["", f"Episodios terminados en el podio (pos. 1-3): **{len(good)}**, {sum(r['seconds'] for r in good) / 3600:.1f} h"]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("\n".join(L))


if __name__ == "__main__":
    main()
