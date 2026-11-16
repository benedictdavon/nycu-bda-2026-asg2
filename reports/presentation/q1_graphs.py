"""Draw the two Graphviz figures for the Q1 report.

Figure 1 (schema graph): the 8 tables, their PKs, FKs and row-time columns.
Figure 2 (entity graph): activity of two cohort users in a 2-day window,
plus one hop of FK targets. CSV files list exactly what Figure 2 draws.

Usage (from any directory; needs the `dot` binary on PATH):
    PATH="<env>/Library/bin:$PATH" python reports/presentation/q1_graphs.py [--data-dir DIR]
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd
from graphviz import Digraph

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (ROOT / "data/legacy/snapshots" / "2053302ba75751be975b9e779f567557b66f4071_20261023"
                    / "assignment2" / "data" / "db")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figures"
FORMATS = ["svg", "png", "pdf"]

# Cohort users: matched on GitHub id, then joined through Users.user_id.
TARGET_GITHUB_IDS = [22996444, 28983449]
# Time window [start, end), timestamps are tz-naive UTC.
WINDOW_START = pd.Timestamp("2026-09-15")
WINDOW_END = pd.Timestamp("2026-09-17")

ENTITY_TITLE = "Relational entity graph: users 22996444 & 28983449, 2026-09-15 to 2026-09-17 UTC"

# One fill color per table, shared by both figures.
COLORS = {
    "Users": "#F4A261",
    "Repositories": "#8AB17D",
    "PullRequests": "#7FB3D5",
    "Issues": "#F7D06A",
    "IssueComments": "#C9B6E4",
    "PRReviews": "#EE8F7A",
    "PRReviewComments": "#9ED9CC",
    "Pushes": "#D0D0D0",
}

# Per table: primary key, row-time column, foreign keys, and columns to load.
# A foreign key is (column, referenced table, referenced PK, known_only_later).
TABLES = {
    "Users": dict(pk="user_id", time="first_seen_at", fks=[],
                  cols=["user_id", "github_user_id", "login", "first_seen_at"]),
    "Repositories": dict(pk="repo_id", time="first_seen_at", fks=[],
                         cols=["repo_id", "name", "first_seen_at"]),
    "PullRequests": dict(pk="pr_id", time="first_seen_at", fks=[
        ("repo_id", "Repositories", "repo_id", False),
        ("author_id", "Users", "user_id", False),
        ("merged_by_id", "Users", "user_id", True),
    ], cols=["pr_id", "number", "title", "repo_id", "author_id", "merged_by_id", "merged_at", "first_seen_at"]),
    "Issues": dict(pk="issue_id", time="first_seen_at", fks=[
        ("repo_id", "Repositories", "repo_id", False),
        ("author_id", "Users", "user_id", False),
        ("assignee_id", "Users", "user_id", False),
        ("pull_request_id", "PullRequests", "pr_id", False),
    ], cols=["issue_id", "number", "title", "repo_id", "author_id", "assignee_id",
             "pull_request_id", "first_seen_at"]),
    "IssueComments": dict(pk="comment_id", time="created_at", fks=[
        ("issue_id", "Issues", "issue_id", False),
        ("repo_id", "Repositories", "repo_id", False),
        ("author_id", "Users", "user_id", False),
    ], cols=["comment_id", "issue_id", "repo_id", "author_id", "created_at"]),
    "PRReviews": dict(pk="review_id", time="first_seen_at", fks=[
        ("pr_id", "PullRequests", "pr_id", False),
        ("repo_id", "Repositories", "repo_id", False),
        ("reviewer_id", "Users", "user_id", False),
    ], cols=["review_id", "pr_id", "repo_id", "reviewer_id", "state", "first_seen_at"]),
    "PRReviewComments": dict(pk="comment_id", time="created_at", fks=[
        ("pr_id", "PullRequests", "pr_id", False),
        ("repo_id", "Repositories", "repo_id", False),
        ("author_id", "Users", "user_id", False),
        ("pull_request_review_id", "PRReviews", "review_id", False),
        ("in_reply_to_id", "PRReviewComments", "comment_id", False),
    ], cols=["comment_id", "pr_id", "repo_id", "author_id", "pull_request_review_id",
             "in_reply_to_id", "created_at"]),
    "Pushes": dict(pk="push_id", time="created_at", fks=[
        ("user_id", "Users", "user_id", False),
        ("repo_id", "Repositories", "repo_id", False),
    ], cols=["push_id", "user_id", "repo_id", "branch", "created_at"]),
}

# Seed rules: (table, actor column). The row-time column is TABLES[table]["time"].
SEED_RULES = [
    ("Pushes", "user_id"),
    ("IssueComments", "author_id"),
    ("PRReviewComments", "author_id"),
    ("PRReviews", "reviewer_id"),
    ("PullRequests", "author_id"),
    ("Issues", "author_id"),
]
EXPECTED_SEEDS = {"Pushes": 2, "IssueComments": 2, "PRReviewComments": 5,
                  "PRReviews": 5, "PullRequests": 2, "Issues": 1}


# ---------- text helpers ----------

def clean(value, limit=None):
    """Make text safe inside a quoted Graphviz label (no quotes, backslashes, newlines)."""
    if value is None or pd.isna(value):
        return ""
    s = " ".join(str(value).split()).replace("\\", "/").replace('"', "'")
    if limit and len(s) > limit:
        s = s[:limit].rstrip() + "..."
    return s


def fmt_time(ts):
    return "n/a" if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def fmt_int(value):
    return "" if pd.isna(value) else str(int(value))


def node_id(table, pk):
    return f"{table}_{pk}"


def render_all(graph, out_dir, stem):
    """Render SVG, PNG (200 dpi) and PDF; the temporary source file is removed."""
    for fmt in FORMATS:
        if fmt == "png":
            graph.graph_attr["dpi"] = "200"
        else:
            graph.graph_attr.pop("dpi", None)
        graph.render(filename=stem, directory=str(out_dir), format=fmt, cleanup=True)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def color_legend_html():
    """HTML-table legend with one color swatch per table."""
    rows = "".join(
        f'<TR><TD BGCOLOR="{color}" WIDTH="16">&#160;</TD><TD ALIGN="LEFT">{table}</TD></TR>'
        for table, color in COLORS.items()
    )
    return ('<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="2" CELLPADDING="3" COLOR="#444444">'
            '<TR><TD COLSPAN="2" ALIGN="LEFT"><B>Table colors</B></TD></TR>' + rows +
            '<TR><TD COLSPAN="2" ALIGN="LEFT">Bold thick border = target user</TD></TR>'
            '</TABLE>>')


# ---------- Figure 1: schema graph ----------

def schema_label(table):
    """HTML-like table label: colored header, PK / FK / time rows, each with a port."""
    spec = TABLES[table]
    rows = [f'<TR><TD BGCOLOR="{COLORS[table]}" ALIGN="CENTER"><B>{table}</B></TD></TR>']
    pk = spec["pk"]
    rows.append(f'<TR><TD PORT="{pk}" ALIGN="LEFT"><B><U>PK</U></B>&#160;{pk}</TD></TR>')
    for col, target, _, later in spec["fks"]:
        note = " (known later)" if later else ""
        rows.append(f'<TR><TD PORT="{col}" ALIGN="LEFT"><I>FK&#160;{col}&#160;'
                    f'&#8594; {target}{note}</I></TD></TR>')
    time_col = spec["time"]
    rows.append(f'<TR><TD PORT="{time_col}" ALIGN="LEFT">time&#160;{time_col}</TD></TR>')
    return ('<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" COLOR="#444444">'
            + "".join(rows) + "</TABLE>>")


def schema_legend_html():
    return ('<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" COLOR="#444444">'
            '<TR><TD ALIGN="LEFT"><B>Legend</B></TD></TR>'
            '<TR><TD ALIGN="LEFT"><B><U>PK</U></B> primary key (row index)</TD></TR>'
            '<TR><TD ALIGN="LEFT"><I>FK</I> foreign key, arrow points to referenced PK</TD></TR>'
            '<TR><TD ALIGN="LEFT">time = row-time column</TD></TR>'
            '<TR><TD ALIGN="LEFT">- - - dashed edge = known only later</TD></TR>'
            '</TABLE>>')


def draw_schema(out_dir):
    g = Digraph("schema", engine="dot")
    g.attr(rankdir="LR", splines="true", nodesep="0.35", ranksep="1.2", fontname="Helvetica",
           label="Schema: 8 tables, keys and foreign keys", labelloc="t", fontsize="16")
    g.attr("node", shape="plaintext", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#333333", arrowsize="0.7")

    for table in TABLES:
        g.node(table, label=schema_label(table))
    g.node("legend", label=schema_legend_html())

    # One edge per foreign key: FK column port -> referenced PK port.
    n_edges = 0
    for table, spec in TABLES.items():
        for col, target, target_pk, later in spec["fks"]:
            attrs = {"style": "dashed", "xlabel": "known later"} if later else {}
            g.edge(f"{table}:{col}:e", f"{target}:{target_pk}:w", **attrs)
            n_edges += 1
    assert n_edges == 20, f"expected 20 FK edges, got {n_edges}"

    render_all(g, out_dir, "q1_schema_graph")
    print(f"Figure 1 (schema): 8 tables, {n_edges} FK edges written.")


# ---------- Figure 2: entity graph ----------

def load_table(data_dir, table):
    spec = TABLES[table]
    df = pd.read_parquet(data_dir / f"{table}.parquet", columns=spec["cols"])
    df = df.set_index(spec["pk"], drop=False)
    assert df.index.is_unique, f"{table} PK is not unique"
    return df


def resolve_targets(users):
    """Map the two GitHub ids to Users.user_id (row index)."""
    hits = users[users["github_user_id"].isin(TARGET_GITHUB_IDS)]
    mapping = dict(zip(hits["github_user_id"].astype(int), hits["user_id"].astype(int)))
    for gid in TARGET_GITHUB_IDS:
        if gid not in mapping:
            raise SystemExit(f"GitHub id {gid} not found in Users")
    return {gid: mapping[gid] for gid in TARGET_GITHUB_IDS}


def entity_lines(table, pk, row):
    """Label lines for one entity row (line 1 = table name + PK)."""
    head = f"{table} {pk}"
    if table == "Users":
        return [head, f"{clean(row['login'])} (github {fmt_int(row['github_user_id'])})"]
    if table == "Repositories":
        return [head, clean(row["name"], 40)]
    if table in ("PullRequests", "Issues"):
        number = fmt_int(row["number"])
        number_text = f"#{number} " if number else ""
        return [head, (number_text + clean(row["title"], 30)).strip()]
    if table == "PRReviews":
        return [head, f"{clean(row['state'])}, {fmt_time(row['first_seen_at'])}"]
    if table == "Pushes":
        return [head, fmt_time(row["created_at"]), f"branch: {clean(row['branch'], 30)}"]
    # IssueComments, PRReviewComments
    return [head, fmt_time(row["created_at"])]


def fk_followable(row, fk):
    """True if the FK is non-null and, for known-later FKs, the PR merged before the window end."""
    col, _, _, known_later = fk
    if pd.isna(row[col]):
        return False
    if known_later:
        return pd.notna(row["merged_at"]) and row["merged_at"] < WINDOW_END
    return True


def add_node(nodes, data, table, pk, is_seed):
    """Add one entity row as a node. Returns False if the row does not exist."""
    pk = int(pk)
    key = (table, pk)
    if key in nodes:
        nodes[key]["is_seed"] = nodes[key]["is_seed"] or is_seed
        return True
    if pk not in data[table].index:
        return False
    row = data[table].loc[pk]
    nodes[key] = {"is_seed": is_seed, "lines": entity_lines(table, pk, row)}
    return True


def build_entity(data, target_pks):
    nodes, unresolved = {}, []
    seed_keys, seed_counts, per_user = [], Counter(), Counter()

    # 1. Seed rows: actor is a target user and the row time is inside the window.
    for table, actor_col in SEED_RULES:
        df = data[table]
        time_col = TABLES[table]["time"]
        mask = (df[actor_col].isin(target_pks.values())
                & (df[time_col] >= WINDOW_START) & (df[time_col] < WINDOW_END))
        hits = df[mask]
        seed_counts[table] = len(hits)
        for pk, actor in zip(hits.index, hits[actor_col]):
            seed_keys.append((table, int(pk)))
            per_user[int(actor)] += 1
            add_node(nodes, data, table, pk, is_seed=True)

    # 2. One hop: follow each non-null FK of each seed row to its referenced row.
    for table, pk in seed_keys:
        row = data[table].loc[pk]
        for fk in TABLES[table]["fks"]:
            if not fk_followable(row, fk):
                continue
            col, target, _, _ = fk
            target_pk = int(row[col])
            if not add_node(nodes, data, target, target_pk, is_seed=False):
                unresolved.append(f"{table}[{pk}].{col} -> {target}[{target_pk}] not found")

    # 3. Edges: any FK whose both ends are already nodes (no further expansion).
    edges, seen = [], set()
    for table, pk in list(nodes):
        row = data[table].loc[pk]
        for fk in TABLES[table]["fks"]:
            col, target, _, _ = fk
            if not fk_followable(row, fk):
                continue
            src, dst = (table, pk), (target, int(row[col]))
            if dst in nodes and (src, dst, col) not in seen:
                seen.add((src, dst, col))
                edges.append((src, dst, col, fk[3]))
    return nodes, edges, seed_counts, per_user, unresolved


def draw_entity(nodes, edges, target_pks, out_dir):
    target_set = set(target_pks.values())
    g = Digraph("entity", engine="dot")
    g.attr(rankdir="LR", splines="true", nodesep="0.25", ranksep="1.0", fontname="Helvetica",
           label=ENTITY_TITLE, labelloc="t", fontsize="18")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10",
           color="#555555")
    g.attr("edge", fontname="Helvetica", fontsize="8", color="#666666", arrowsize="0.7")

    for (table, pk), info in nodes.items():
        attrs = {"fillcolor": COLORS[table]}
        if table == "Users" and pk in target_set:
            attrs.update(penwidth="3.5", color="#000000", fontname="Helvetica-Bold", fontsize="13")
        g.node(node_id(table, pk), label="\\n".join(info["lines"]), **attrs)

    for src, dst, col, later in edges:
        attrs = {"label": f"{col} (known later)", "style": "dashed"} if later else {"label": col}
        g.edge(node_id(*src), node_id(*dst), **attrs)

    g.node("legend", shape="plaintext", label=color_legend_html())
    render_all(g, out_dir, "q1_entity_graph")


def main():
    parser = argparse.ArgumentParser(description="Draw the Q1 schema and entity graphs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="folder with the 8 parquet files and SCHEMA.md")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    draw_schema(args.out_dir)

    data = {table: load_table(args.data_dir, table) for table in TABLES}
    target_pks = resolve_targets(data["Users"])
    login_of = {int(pk): data["Users"].loc[pk, "login"] for pk in target_pks.values()}
    print("Target users:", {gid: f"user_id {pk} ({login_of[pk]})" for gid, pk in target_pks.items()})

    nodes, edges, seed_counts, per_user, unresolved = build_entity(data, target_pks)

    # Seed counts, checked against the expected numbers.
    print("\nSeed rows per table:", dict(seed_counts), "total", sum(seed_counts.values()))
    print("Seed rows per user:", {f"{pk} ({login_of[pk]})": n for pk, n in per_user.items()})
    mismatch = {t: (seed_counts[t], e) for t, e in EXPECTED_SEEDS.items() if seed_counts[t] != e}
    if mismatch:
        print("SEED COUNTS DIFFER from expected (actual, expected):", mismatch)
    else:
        print("Seed counts match the expected 17 rows.")

    node_totals = Counter(table for table, _ in nodes)
    print("\nNodes per table:", dict(node_totals), "total", len(nodes))
    print("Edges:", len(edges))
    print("Unresolved FKs:", len(unresolved))
    for line in unresolved:
        print("  ", line)

    draw_entity(nodes, edges, target_pks, args.out_dir)

    write_csv(args.out_dir / "q1_entity_nodes.csv",
              ["node_id", "table", "pk", "label", "is_seed"],
              [(node_id(t, pk), t, pk, " | ".join(info["lines"]), info["is_seed"])
               for (t, pk), info in nodes.items()])
    write_csv(args.out_dir / "q1_entity_edges.csv",
              ["src", "dst", "fk_column"],
              [(node_id(*src), node_id(*dst), col) for src, dst, col, _ in edges])

    print("\nFigure 2 (entity): rendered svg, png (200 dpi), pdf; CSVs written.")
    print("Output folder:", args.out_dir)


if __name__ == "__main__":
    main()
