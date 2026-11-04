#!/usr/bin/env python3
"""
Merge two data.db files from the keystroke-emotion app into one clean database.

    python merge_databases.py "C:\\path\\to\\A\\data.db" "C:\\path\\to\\B\\data.db" merged.db

What it does
  * Never changes the input files (they are opened read-only).
  * Drops participants with no typing samples (abandoned attempts).
  * Treats the same name (ignoring case and extra spaces) as the same person.
    If both files hold identical data for that person, one copy is kept.
    If the data differs, it stops and lists the conflict; re-run with
    --on-conflict to choose what to do.
  * Renumbers participants, sessions, samples and keystroke events so ids from
    the two systems cannot collide, then checks the result against the sources.
Only the standard library is needed. A report is also saved as <out>.report.txt
(it contains participant names, so treat it like the database itself).
"""
import argparse
import hashlib
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REQUIRED = ["participants", "emotion_sessions", "typing_samples", "keystroke_events"]
OPTIONAL = ["comprehension_checks"]  # only present in databases from older code
LOG = []


def say(text=""):
    print(text)
    LOG.append(text)


def open_ro(path):
    p = Path(path).resolve()
    if not p.exists():
        sys.exit(f"File not found: {path}")
    con = sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def has_table(con, name):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def table_columns(con, table):
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]


def norm(name):
    return " ".join((name or "").split()).casefold()


def val(row, key):
    return row[key] if key in row.keys() else None


def summaries(con):
    """Per participant id: sample count, event count and a content fingerprint."""
    events = {r[0]: r[1] for r in con.execute(
        "SELECT typing_sample_id, COUNT(*) FROM keystroke_events GROUP BY typing_sample_id")}
    by_pid = defaultdict(list)
    for s in con.execute("SELECT * FROM typing_samples ORDER BY id"):
        by_pid[s["participant_id"]].append(s)
    out = {}
    for pid, rows in by_pid.items():
        items = sorted(
            (str(val(s, "sample_type")), str(val(s, "label")), str(val(s, "self_reported_emotion")),
             str(val(s, "text_content")), str(val(s, "word_count")), str(events.get(s["id"], 0)))
            for s in rows)
        out[pid] = {
            "n_samples": len(rows),
            "n_events": sum(events.get(s["id"], 0) for s in rows),
            "fp": hashlib.sha1(repr(items).encode("utf-8")).hexdigest(),
        }
    return out


def load_source(order, label, path):
    con = open_ro(path)
    for t in REQUIRED:
        if not has_table(con, t):
            sys.exit(f"{path}: table '{t}' not found. Is this the right file?")
    src = {"order": order, "label": label, "path": path, "con": con}
    summ = summaries(con)
    people = []
    for p in con.execute("SELECT * FROM participants ORDER BY id"):
        s = summ.get(p["id"], {"n_samples": 0, "n_events": 0, "fp": "none"})
        people.append({"src": src, "id": p["id"], "name": p["name"], "key": norm(p["name"]),
                       "row": p, **s})
    return src, people


def choose(people, policy):
    kept, dropped, conflicts = [], [], []
    groups = defaultdict(list)
    for p in people:
        if p["n_samples"] == 0:
            dropped.append((p, "no typing samples"))
        else:
            groups[p["key"]].append(p)
    for items in groups.values():
        items.sort(key=lambda p: (p["src"]["order"], p["id"]))
        first = items[0]
        if len(items) == 1:
            kept.append(first)
        elif len({p["fp"] for p in items}) == 1:
            kept.append(first)
            for p in items[1:]:
                dropped.append((p, f"identical copy of file {first['src']['label']} id {first['id']}"))
        else:
            conflicts.append(items)
            if policy == "keep-all":
                kept.extend(items)
            elif policy == "keep-first":
                kept.append(first)
                dropped.extend((p, "conflict: kept the first file's version") for p in items[1:])
            elif policy == "keep-larger":
                best = max(items, key=lambda p: (p["n_samples"], p["n_events"], -p["src"]["order"]))
                kept.append(best)
                dropped.extend((p, "conflict: kept the version with more data") for p in items if p is not best)
    kept.sort(key=lambda p: (p["src"]["order"], p["id"]))
    return kept, dropped, conflicts


def copy_rows(out, table, rows, tcols, fixed=None, skip=()):
    fixed = fixed or {}
    if not rows:
        return
    cols = [c for c in tcols[table] if c not in skip and (c in fixed or c in rows[0].keys())]
    q = ",".join(f'"{c}"' for c in cols)
    sql = f'INSERT INTO "{table}" ({q}) VALUES ({",".join("?" * len(cols))})'
    out.executemany(sql, [[fixed[c] if c in fixed else r[c] for c in cols] for r in rows])


def build(out_path, sources, kept):
    out = sqlite3.connect(out_path)
    out.execute("PRAGMA foreign_keys = OFF")
    tcols = {}
    for t in REQUIRED + OPTIONAL:
        for s in sources:
            if has_table(s["con"], t):
                ddl = s["con"].execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()[0]
                out.execute(ddl)
                tcols[t] = table_columns(out, t)
                break
    new_sid = new_samp = 0
    for new_pid, p in enumerate(kept, start=1):
        con = p["src"]["con"]
        copy_rows(out, "participants", [p["row"]], tcols, {"id": new_pid})
        sess_map = {}
        for sess in con.execute("SELECT * FROM emotion_sessions WHERE participant_id=? ORDER BY id", (p["id"],)).fetchall():
            new_sid += 1
            sess_map[sess["id"]] = new_sid
            copy_rows(out, "emotion_sessions", [sess], tcols, {"id": new_sid, "participant_id": new_pid})
        for smp in con.execute("SELECT * FROM typing_samples WHERE participant_id=? ORDER BY id", (p["id"],)).fetchall():
            if smp["session_id"] not in sess_map:
                sys.exit(f"{p['src']['path']}: sample {smp['id']} points to a session that does not belong to "
                         f"participant {p['id']} ({p['name']}). Fix or exclude this participant first.")
            new_samp += 1
            copy_rows(out, "typing_samples", [smp], tcols,
                      {"id": new_samp, "participant_id": new_pid, "session_id": sess_map[smp["session_id"]]})
            ev = con.execute("SELECT * FROM keystroke_events WHERE typing_sample_id=? ORDER BY sequence_index, id",
                             (smp["id"],)).fetchall()
            copy_rows(out, "keystroke_events", ev, tcols, {"typing_sample_id": new_samp}, skip=("id",))
            if "comprehension_checks" in tcols and has_table(con, "comprehension_checks"):
                cc = con.execute("SELECT * FROM comprehension_checks WHERE typing_sample_id=?", (smp["id"],)).fetchall()
                copy_rows(out, "comprehension_checks", cc, tcols, {"typing_sample_id": new_samp}, skip=("id",))
    out.commit()
    out.close()


def verify(out_path, kept):
    con = sqlite3.connect(out_path)
    con.row_factory = sqlite3.Row
    problems = []
    if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        problems.append("SQLite integrity_check failed")
    if con.execute("PRAGMA foreign_key_check").fetchall():
        problems.append("foreign key violations found")
    merged = summaries(con)
    for new_pid, p in enumerate(kept, start=1):
        m = merged.get(new_pid)
        if not m or m["fp"] != p["fp"]:
            problems.append(f"data for '{p['name']}' does not match its source after merging")
    names = [norm(r[0]) for r in con.execute("SELECT name FROM participants")]
    if len(names) != len(set(names)):
        problems.append("duplicate names remain (expected only with --on-conflict keep-all)")
    return con, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db_a", help="first database (preferred when copies are identical)")
    ap.add_argument("db_b", help="second database")
    ap.add_argument("out", help="merged database to create")
    ap.add_argument("--on-conflict", choices=["abort", "keep-first", "keep-larger", "keep-all"], default="abort",
                    help="what to do when the same name has different data in the two files (default: abort)")
    ap.add_argument("--force", action="store_true", help="overwrite the output file if it exists")
    args = ap.parse_args()

    in_paths = {str(Path(p).resolve()) for p in (args.db_a, args.db_b)}
    if str(Path(args.out).resolve()) in in_paths:
        sys.exit("The output file must be different from the input files.")
    if os.path.exists(args.out) and not args.force:
        sys.exit(f"{args.out} already exists. Use --force to overwrite it.")

    src_a, people_a = load_source(0, "A", args.db_a)
    src_b, people_b = load_source(1, "B", args.db_b)
    kept, dropped, conflicts = choose(people_a + people_b, args.on_conflict)

    say(f"A: {len(people_a)} participant rows | B: {len(people_b)} participant rows")
    say("\nDropped:")
    for p, why in dropped:
        say(f"  file {p['src']['label']} id {p['id']:>3}  {p['name']}  ({why})")
    if conflicts:
        say("\nCONFLICTS (same name, different data):")
        for items in conflicts:
            for p in items:
                say(f"  file {p['src']['label']} id {p['id']:>3}  {p['name']}  "
                    f"{p['n_samples']} samples, {p['n_events']} events")
        if args.on_conflict == "abort":
            say("\nNothing was written. Decide which version is right, then re-run with "
                "--on-conflict keep-first | keep-larger | keep-all")
            sys.exit(1)

    if os.path.exists(args.out):
        os.remove(args.out)
    build(args.out, [src_a, src_b], kept)
    con, problems = verify(args.out, kept)

    say(f"\nMerged database: {args.out}")
    say(f"{'id':>3}  {'name':<24} {'from':<4} {'old id':>6} {'samples':>7} {'key_presses':>11}")
    rows = con.execute("""SELECT p.id, p.name, COUNT(DISTINCT t.id) AS samples,
                                 SUM(CASE WHEN k.event_type='down' THEN 1 ELSE 0 END) AS presses
                          FROM participants p
                          LEFT JOIN typing_samples t ON t.participant_id = p.id
                          LEFT JOIN keystroke_events k ON k.typing_sample_id = t.id
                          GROUP BY p.id, p.name ORDER BY p.id""").fetchall()
    for r, p in zip(rows, kept):
        flag = "" if r["samples"] == 10 else "   <-- not 10 samples"
        say(f"{r['id']:>3}  {r['name']:<24} {p['src']['label']:<4} {p['id']:>6} {r['samples']:>7} {r['presses'] or 0:>11}{flag}")
    totals = con.execute("""SELECT (SELECT COUNT(*) FROM participants), (SELECT COUNT(*) FROM typing_samples),
                                   (SELECT COUNT(*) FROM keystroke_events)""").fetchone()
    say(f"\nTotals: {totals[0]} participants, {totals[1]} samples, {totals[2]} keystroke events")

    say("\nOpposite-emotion assignments (self-reported -> asked to write about):")
    for r in con.execute("""SELECT self_reported_emotion AS s, target_emotion AS t, COUNT(*) AS n
                            FROM typing_samples WHERE sample_type='opposite_response'
                            GROUP BY 1, 2 ORDER BY 1, 2"""):
        say(f"  {r['s']:<6} -> {r['t']:<6} {r['n']}")

    if problems:
        say("\nPROBLEMS:")
        for pr in problems:
            say(f"  - {pr}")
    else:
        say("\nChecks passed: integrity OK, no foreign-key problems, every participant's data matches its source.")
    Path(args.out + ".report.txt").write_text("\n".join(LOG), encoding="utf-8")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
