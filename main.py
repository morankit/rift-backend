"""
money_muling_engine.py — RIFT 2026 Hackathon Submission
Graph-Based Financial Crime Detection Engine
"""

import io
import logging
import math
import time
import uuid
import concurrent.futures
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("muling_engine")


class Config:
    REQUIRED_COLUMNS: Set[str] = {
        "transaction_id", "sender_id", "receiver_id", "amount", "timestamp"
    }
    ID_COLUMNS: Tuple[str, ...] = ("transaction_id", "sender_id", "receiver_id")

    CYCLE_MIN_LEN: int = 3
    CYCLE_MAX_LEN: int = 5

    SMURF_MIN_COUNTERPARTIES: int = 10
    SMURF_WINDOW_HOURS: int = 72

    SHELL_MIN_HOPS: int = 3
    SHELL_MAX_TX_PER_NODE: int = 3
    SHELL_MAX_DEPTH: int = 8

    MERCHANT_PERCENTILE: float = 97.0
    MERCHANT_MIN_TX: int = 50

    W_CYCLE:  float = 0.40
    W_SMURF:  float = 0.30
    W_SHELL:  float = 0.15
    W_VOLUME: float = 0.15

    VOLUME_LOG_SCALE: float = 1_000_000.0


CFG = Config()


def _canonical_cycle(cycle: List[str]) -> Tuple[str, ...]:
    min_i = cycle.index(min(cycle))
    return tuple(cycle[min_i:] + cycle[:min_i])


def _log_volume_score(volume: float, scale: float) -> float:
    if volume <= 0:
        return 0.0
    return min(math.log1p(volume) / math.log1p(scale), 1.0)


def _ring_id(index: int) -> str:
    return f"RING_{index + 1:03d}"


def _build_merchant_whitelist(count_map: Dict[str, int]) -> Set[str]:
    if not count_map:
        return set()
    counts = sorted(count_map.values())
    threshold_idx = int(len(counts) * CFG.MERCHANT_PERCENTILE / 100)
    percentile_val = counts[min(threshold_idx, len(counts) - 1)]
    threshold = max(CFG.MERCHANT_MIN_TX, percentile_val)
    return {acct for acct, cnt in count_map.items() if cnt >= threshold}


def detect_circular_routing(G: nx.MultiDiGraph) -> List[List[str]]:
    simple_G = nx.DiGraph(G)
    changed = True
    while changed:
        low = [n for n in simple_G if simple_G.degree(n) < 2]
        simple_G.remove_nodes_from(low)
        changed = bool(low)

    cycles: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = set()

    for cycle in nx.simple_cycles(simple_G):
        if CFG.CYCLE_MIN_LEN <= len(cycle) <= CFG.CYCLE_MAX_LEN:
            key = _canonical_cycle(cycle)
            if key not in seen:
                seen.add(key)
                cycles.append(cycle)

    logger.info("Cycles: %d unique rings found", len(cycles))
    return cycles


def _sliding_window_check(
    grp: pd.DataFrame,
    counterpart_col: str,
    window_hours: int,
    threshold: int,
) -> Tuple[bool, int, float, str]:
    grp = grp.reset_index(drop=True)
    left = 0
    cp_counts: Dict[str, int] = {}
    window_amt = 0.0

    for right in range(len(grp)):
        cp  = grp.at[right, counterpart_col]
        amt = grp.at[right, "amount"]
        cp_counts[cp] = cp_counts.get(cp, 0) + 1
        window_amt += amt

        cutoff = grp.at[right, "timestamp"] - pd.Timedelta(hours=window_hours)
        while grp.at[left, "timestamp"] < cutoff:
            old_cp = grp.at[left, counterpart_col]
            cp_counts[old_cp] -= 1
            if cp_counts[old_cp] == 0:
                del cp_counts[old_cp]
            window_amt -= grp.at[left, "amount"]
            left += 1

        if len(cp_counts) >= threshold:
            return True, len(cp_counts), round(window_amt, 2), str(grp.at[left, "timestamp"])

    return False, 0, 0.0, ""


def detect_smurfing(
    df: pd.DataFrame,
    whitelist: Set[str],
) -> Dict[str, Dict[str, Any]]:
    df_t = df[["receiver_id", "sender_id", "amount", "timestamp"]].copy()
    df_t["timestamp"] = pd.to_datetime(df_t["timestamp"], utc=True, errors="coerce")
    dropped = df_t["timestamp"].isna().sum()
    if dropped:
        logger.warning("Smurfing: %d rows with unparseable timestamps dropped", dropped)
    df_t = df_t.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    flagged: Dict[str, Dict[str, Any]] = {}

    for receiver, grp in df_t.groupby("receiver_id", sort=False):
        if str(receiver) in whitelist:
            continue
        found, cnt, amt, ws = _sliding_window_check(
            grp, "sender_id", CFG.SMURF_WINDOW_HOURS, CFG.SMURF_MIN_COUNTERPARTIES
        )
        if found:
            flagged[str(receiver)] = {"pattern": "fan_in", "fan_count": cnt, "amount": amt, "window_start": ws}

    for sender, grp in df_t.groupby("sender_id", sort=False):
        if str(sender) in whitelist or str(sender) in flagged:
            continue
        found, cnt, amt, ws = _sliding_window_check(
            grp, "receiver_id", CFG.SMURF_WINDOW_HOURS, CFG.SMURF_MIN_COUNTERPARTIES
        )
        if found:
            flagged[str(sender)] = {"pattern": "fan_out", "fan_count": cnt, "amount": amt, "window_start": ws}

    logger.info("Smurfing: %d accounts flagged", len(flagged))
    return flagged


def detect_layered_shells(
    G: nx.MultiDiGraph,
    count_map: Dict[str, int],
    whitelist: Set[str],
) -> List[List[str]]:
    def is_shell_interior(node: str) -> bool:
        return (
            node not in whitelist
            and count_map.get(str(node), 0) <= CFG.SHELL_MAX_TX_PER_NODE
        )

    candidate_nodes = {n for n in G.nodes() if str(n) not in whitelist}
    sub_G: nx.DiGraph = nx.DiGraph(G.subgraph(candidate_nodes))

    chains: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = set()

    def dfs(node: str, path: List[str]) -> None:
        if len(path) >= CFG.SHELL_MIN_HOPS:
            interior = path[1:-1]
            if interior and all(is_shell_interior(n) for n in interior):
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    chains.append(list(path))

        if len(path) < CFG.SHELL_MAX_DEPTH:
            for succ in sub_G.successors(node):
                if succ not in path:
                    path.append(succ)
                    dfs(succ, path)
                    path.pop()

    chain_heads = [n for n in sub_G if sub_G.in_degree(n) == 0]
    if not chain_heads:
        chain_heads = list(sub_G.nodes())

    for head in chain_heads:
        dfs(head, [head])

    logger.info("Shell chains: %d found", len(chains))
    return chains


def _detected_patterns(
    in_cycle: bool,
    cycle_len: int,
    smurf_info: Dict[str, Any],
    in_shell: bool,
    volume: float,
) -> List[str]:
    patterns = []
    if in_cycle and cycle_len:
        patterns.append(f"cycle_length_{cycle_len}")
    if smurf_info:
        pat = smurf_info.get("pattern", "fan_in")
        patterns.append("high_velocity" if pat == "fan_in" else "fan_out")
    if in_shell:
        patterns.append("layered_shell")
    if volume > 500_000:
        patterns.append("high_volume")
    return sorted(patterns)


def _validate(df: pd.DataFrame) -> None:
    missing = CFG.REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("CSV contains no data rows.")


def run_full_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    _validate(df)
    start = time.perf_counter()

    in_vol  = df.groupby("receiver_id")["amount"].sum()
    out_vol = df.groupby("sender_id")["amount"].sum()
    vol_map: Dict[str, float] = in_vol.add(out_vol, fill_value=0).to_dict()

    in_cnt  = df.groupby("receiver_id")["transaction_id"].count()
    out_cnt = df.groupby("sender_id")["transaction_id"].count()
    cnt_map: Dict[str, int] = in_cnt.add(out_cnt, fill_value=0).astype(int).to_dict()

    whitelist = _build_merchant_whitelist(cnt_map)
    logger.info("Merchant whitelist: %d accounts", len(whitelist))

    G = nx.MultiDiGraph()
    G.add_edges_from([
        (str(r.sender_id), str(r.receiver_id), {"amount": r.amount})
        for r in df.itertuples()
    ])
    total_accounts = G.number_of_nodes()
    logger.info("Graph: %d nodes, %d edges", total_accounts, G.number_of_edges())

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_cycles = pool.submit(detect_circular_routing, G)
        f_smurf  = pool.submit(detect_smurfing, df, whitelist)
        f_shells = pool.submit(detect_layered_shells, G, cnt_map, whitelist)
        cycles       = f_cycles.result()
        smurf_map    = f_smurf.result()
        shell_chains = f_shells.result()

    ring_counter = 0
    fraud_rings_raw: List[Dict[str, Any]] = []
    acct_to_ring: Dict[str, str] = {}
    acct_to_cycle_len: Dict[str, int] = {}

    for cycle in cycles:
        rid = _ring_id(ring_counter); ring_counter += 1
        fraud_rings_raw.append({
            "ring_id": rid, "member_accounts": list(cycle),
            "pattern_type": "cycle",
        })
        for acct in cycle:
            acct_to_ring.setdefault(acct, rid)
            acct_to_cycle_len[acct] = len(cycle)

    for chain in shell_chains:
        rid = _ring_id(ring_counter); ring_counter += 1
        fraud_rings_raw.append({
            "ring_id": rid, "member_accounts": list(chain),
            "pattern_type": "layered_shells",
        })
        for acct in chain[1:-1]:
            acct_to_ring.setdefault(acct, rid)

    for acct in smurf_map:
        if acct not in acct_to_ring:
            rid = _ring_id(ring_counter); ring_counter += 1
            fraud_rings_raw.append({
                "ring_id": rid, "member_accounts": [acct],
                "pattern_type": "smurfing",
            })
            acct_to_ring[acct] = rid

    cycle_nodes = set(acct_to_cycle_len.keys())
    shell_nodes = {n for c in shell_chains if len(c) > 2 for n in c[1:-1]}
    all_flagged = cycle_nodes | set(smurf_map.keys()) | shell_nodes

    score_map: Dict[str, float] = {}
    suspicious_out: List[Dict[str, Any]] = []

    for acct in all_flagged:
        in_cycle = acct in cycle_nodes
        in_smurf = acct in smurf_map
        in_shell = acct in shell_nodes
        vol      = vol_map.get(acct, 0.0)

        score = round(min(
            CFG.W_CYCLE  * (100.0 if in_cycle else 0.0) +
            CFG.W_SMURF  * (100.0 if in_smurf else 0.0) +
            CFG.W_SHELL  * (100.0 if in_shell else 0.0) +
            CFG.W_VOLUME * _log_volume_score(vol, CFG.VOLUME_LOG_SCALE) * 100.0,
            100.0,
        ), 2)

        score_map[acct] = score

        suspicious_out.append({
            "account_id":        acct,
            "suspicion_score":   score,
            "detected_patterns": _detected_patterns(
                in_cycle, acct_to_cycle_len.get(acct, 0),
                smurf_map.get(acct, {}), in_shell, vol,
            ),
            "ring_id": acct_to_ring.get(acct, "NONE"),
        })

    suspicious_out.sort(key=lambda x: x["suspicion_score"], reverse=True)

    fraud_rings_out: List[Dict[str, Any]] = []
    for ring in fraud_rings_raw:
        members = ring["member_accounts"]
        avg_score = (
            round(sum(score_map.get(m, 0.0) for m in members) / len(members), 1)
            if members else 0.0
        )
        fraud_rings_out.append({
            "ring_id":         ring["ring_id"],
            "member_accounts": members,
            "pattern_type":    ring["pattern_type"],
            "risk_score":      avg_score,
        })

    # ── Build graph_data for the frontend ─────────────────────────────────────
    # Include all edges where at least one endpoint is a flagged/ring account,
    # so the frontend RingGraph can draw real transaction arrows.
    flagged_set = {a["account_id"] for a in suspicious_out}
    graph_edges = []
    seen_edges: Set[Tuple[str, str]] = set()
    for _, row in df.iterrows():
        s, d = str(row["sender_id"]), str(row["receiver_id"])
        if (s in flagged_set or d in flagged_set) and (s, d) not in seen_edges:
            seen_edges.add((s, d))
            graph_edges.append({"source": s, "target": d})

    elapsed = round(time.perf_counter() - start, 4)
    logger.info("Done %.4fs — %d suspicious, %d rings", elapsed, len(suspicious_out), len(fraud_rings_out))

    return {
        "detection": {
            "suspicious_accounts": suspicious_out,
            "fraud_rings":         fraud_rings_out,
        },
        "graph_data": {
            "edges": graph_edges,
        },
        "summary": {
            "total_accounts_analyzed":     total_accounts,
            "suspicious_accounts_flagged": len(suspicious_out),
            "fraud_rings_detected":        len(fraud_rings_out),
            "processing_time_seconds":     elapsed,
        },
    }


# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(title="RIFT 2026 — Money Muling Detection Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory result store: processing_id → full result dict
_results: Dict[str, Any] = {}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accepts a CSV, runs full analysis immediately, stores results keyed by
    processing_id, and returns processing_id + summary so the frontend can
    fetch /results/{processing_id} next.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV files accepted.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}") from exc

    for col in CFG.ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    missing = CFG.REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {sorted(missing)}")

    try:
        result = run_full_analysis(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    processing_id = str(uuid.uuid4())
    _results[processing_id] = result

    logger.info("Stored results under processing_id=%s", processing_id)

    return JSONResponse(content={
        "processing_id": processing_id,
        "summary": result["summary"],
    })


@app.get("/results/{processing_id}")
async def get_results(processing_id: str) -> JSONResponse:
    """
    Returns the full analysis payload:
      { detection: { suspicious_accounts, fraud_rings }, graph_data: { edges } }
    """
    result = _results.get(processing_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for processing_id '{processing_id}'. "
                   "Upload a CSV first via POST /upload.",
        )
    return JSONResponse(content=result)


@app.get("/download/{processing_id}")
async def download_json(processing_id: str) -> JSONResponse:
    """Returns the full result as a downloadable JSON attachment."""
    result = _results.get(processing_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Results not found.")
    return JSONResponse(
        content=result,
        headers={"Content-Disposition": "attachment; filename=muling_analysis.json"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


    from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

app = FastAPI()

app.mount("/", StaticFiles(directory="dist", html=True), name="static")