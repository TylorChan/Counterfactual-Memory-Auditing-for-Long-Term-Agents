#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longmemeval_counterfactual import append_cf_outputs, build_cf_specs, summarize_replay_cf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CF-only replays from existing baseline audit artifacts.")
    parser.add_argument("--agent", required=True, choices=("anna", "share", "memoryos", "ldagent", "theanine", "mem0"))
    parser.add_argument("--longmemeval-file", type=Path, required=True)
    parser.add_argument("--baseline-trace-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-audit-queries", type=Path, default=None)
    parser.add_argument("--baseline-audit-writes", type=Path, default=None)
    parser.add_argument("--cf-tag", type=str, required=True)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai-base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--cf-target-scope", choices=("prompt", "retrieved", "candidate"), default="prompt")
    parser.add_argument("--cf-max-writes", type=int, default=8)
    parser.add_argument("--cf-dominance-threshold", type=float, default=0.75)
    parser.add_argument("--cf-rule-mode", choices=("all", "rollback-only"), default="all")
    parser.add_argument("--share-dir", type=Path, default=REPO_ROOT / "SHARE")
    parser.add_argument("--memoryos-dir", type=Path, default=REPO_ROOT / "MemoryOS")
    parser.add_argument("--ld-agent-dir", type=Path, default=REPO_ROOT / "LD-Agent")
    parser.add_argument("--anna-agent-dir", type=Path, default=REPO_ROOT / "AnnaAgent")
    parser.add_argument("--theanine-dir", type=Path, default=REPO_ROOT / "Theanine")
    parser.add_argument("--mem0-dir", type=Path, default=REPO_ROOT / "mem0")
    parser.add_argument("--runtime-dir", type=Path, default=REPO_ROOT / "cf_only_runtime")
    return parser.parse_args()


def _parse_module_args(parse_fn, argv: Sequence[str]):
    old_argv = sys.argv[:]
    try:
        sys.argv = ["cf_only"] + list(argv)
        return parse_fn()
    finally:
        sys.argv = old_argv


def _derive_audit_paths(trace_path: Path) -> Tuple[Path, Path]:
    stem = trace_path.name[:-6] if trace_path.name.endswith(".jsonl") else trace_path.name
    return (
        trace_path.with_name(f"{stem}.audit_queries.jsonl"),
        trace_path.with_name(f"{stem}.audit_writes.jsonl"),
    )


def _derive_cf_only_paths(trace_path: Path, cf_tag: str) -> Tuple[Path, Path]:
    stem = trace_path.name[:-6] if trace_path.name.endswith(".jsonl") else trace_path.name
    return (
        trace_path.with_name(f"{stem}.{cf_tag}.cf_runs.jsonl"),
        trace_path.with_name(f"{stem}.{cf_tag}.cf_queries.jsonl"),
    )


def _load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_dataset(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset in {path}")
    return {str(item["question_id"]): item for item in data}


def _group_writes_by_qid(records: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get("question_id")), []).append(record)
    return grouped


def _prepare_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.touch()


def _iter_baseline_queries(path: Path) -> List[Dict]:
    records = _load_jsonl(path)
    return [r for r in records if r.get("trace_kind") == "baseline_query"]


def _filter_specs(specs: Sequence, rule_mode: str) -> List:
    if rule_mode == "rollback-only":
        return [spec for spec in specs if getattr(spec, "rule_id", "") == "rollback_skip"]
    return list(specs)


def run_share(args: argparse.Namespace, entries: Dict[str, Dict], query_records: List[Dict], writes_by_qid: Dict[str, List[Dict]], run_path: Path, query_path: Path) -> None:
    import share_longmemeval_bridge.run_infer as share

    share_args = _parse_module_args(
        share.parse_args,
        [
            "--share-dir", str(args.share_dir),
            "--longmemeval-file", str(args.longmemeval_file),
            "--out-jsonl", str(args.runtime_dir / "dummy_share.jsonl"),
            "--trace-jsonl", str(args.runtime_dir / "dummy_share.trace.jsonl"),
            "--openai-base-url", args.openai_base_url,
            "--llm-model", args.llm_model,
        ],
    )
    api_key = share_args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for SHARE CF-only run.")
    llm = share.OpenAIJsonClient(
        api_key=api_key,
        model=share_args.llm_model,
        base_url=share_args.openai_base_url,
        temperature=share_args.temperature,
        max_tokens=share_args.max_tokens,
        timeout=share_args.timeout,
    )
    pbar = tqdm(query_records, total=len(query_records), desc="SHARE CF-only", unit="q")
    for idx, baseline_qr in enumerate(pbar, start=1):
        qid = str(baseline_qr["question_id"])
        entry = entries[qid]
        write_records = writes_by_qid.get(qid, [])
        events = share.collect_share_write_events(entry, share_args.preserve_session_order, share_args.max_session_dialogue_chars)
        specs = _filter_specs(build_cf_specs(
            question_type=str(baseline_qr.get("question_type") or entry.get("question_type") or "unknown"),
            query_record=baseline_qr,
            write_records=write_records,
            answer_session_ids=entry.get("answer_session_ids", []),
            max_writes=args.cf_max_writes,
            scope=args.cf_target_scope,
        ), args.cf_rule_mode)
        cf_results = []
        for spec in specs:
            mutated_events, target_timestamp = share.apply_share_cf_spec(events, spec)
            outcome = share.run_share_replay(entry=entry, events=mutated_events, llm=llm, args=share_args)
            cf_results.append({
                "spec": spec,
                "cf_answer": outcome["hypothesis"],
                "cf_retrieved_write_ids": outcome["query_record"].get("retrieved_write_ids", []),
                "cf_prompt_write_ids": outcome["query_record"].get("prompt_write_ids", []),
                "target_timestamp": target_timestamp,
            })
        run_records, query_summary = summarize_replay_cf(
            agent="share",
            entry=entry,
            baseline_query_record=baseline_qr,
            write_records=write_records,
            cf_results=cf_results,
            dominance_threshold=args.cf_dominance_threshold,
        )
        append_cf_outputs(run_path=run_path, query_path=query_path, run_records=run_records, query_summary=query_summary)


def run_memoryos(args: argparse.Namespace, entries: Dict[str, Dict], query_records: List[Dict], writes_by_qid: Dict[str, List[Dict]], run_path: Path, query_path: Path) -> None:
    import memos_longmemeval_bridge.run_infer as memos

    memoryos_args = _parse_module_args(
        memos.parse_args,
        [
            "--memoryos-dir", str(args.memoryos_dir),
            "--longmemeval-file", str(args.longmemeval_file),
            "--out-jsonl", str(args.runtime_dir / "dummy_memoryos.jsonl"),
            "--trace-jsonl", str(args.runtime_dir / "dummy_memoryos.trace.jsonl"),
            "--openai-base-url", args.openai_base_url,
            "--llm-model", args.llm_model,
            "--reset-mode", "reinit",
        ],
    )
    silence_memoryos_logs = not memoryos_args.verbose_memoryos
    per_sample_root = args.runtime_dir / "memoryos_cf_only"
    per_sample_root.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(query_records, total=len(query_records), desc="MemoryOS CF-only", unit="q")
    for idx, baseline_qr in enumerate(pbar, start=1):
        qid = str(baseline_qr["question_id"])
        entry = entries[qid]
        write_records = writes_by_qid.get(qid, [])
        events = memos.collect_memoryos_write_events(entry, memoryos_args.preserve_session_order)
        specs = _filter_specs(build_cf_specs(
            question_type=str(baseline_qr.get("question_type") or entry.get("question_type") or "unknown"),
            query_record=baseline_qr,
            write_records=write_records,
            answer_session_ids=entry.get("answer_session_ids", []),
            max_writes=args.cf_max_writes,
            scope=args.cf_target_scope,
        ), args.cf_rule_mode)
        cf_results = []
        for spec in specs:
            mutated_events, target_timestamp = memos.apply_memoryos_cf_spec(events, spec)
            outcome = memos.run_memoryos_replay(
                entry=entry,
                events=mutated_events,
                args=memoryos_args,
                sample_storage_root=per_sample_root,
                sample_tag=f"{idx:03d}_{qid}_{spec.rule_id}_{spec.target_write_id[:8]}",
                silence_memoryos_logs=silence_memoryos_logs,
            )
            cf_results.append({
                "spec": spec,
                "cf_answer": outcome["hypothesis"],
                "cf_retrieved_write_ids": outcome["query_record"].get("retrieved_write_ids", []),
                "cf_prompt_write_ids": outcome["query_record"].get("prompt_write_ids", []),
                "target_timestamp": target_timestamp,
            })
        run_records, query_summary = summarize_replay_cf(
            agent="memoryos",
            entry=entry,
            baseline_query_record=baseline_qr,
            write_records=write_records,
            cf_results=cf_results,
            dominance_threshold=args.cf_dominance_threshold,
        )
        append_cf_outputs(run_path=run_path, query_path=query_path, run_records=run_records, query_summary=query_summary)


def run_mem0(args: argparse.Namespace, entries: Dict[str, Dict], query_records: List[Dict], writes_by_qid: Dict[str, List[Dict]], run_path: Path, query_path: Path) -> None:
    import mem0_longmemeval_bridge.run_infer as mem0_bridge

    mem0_args = _parse_module_args(
        mem0_bridge.parse_args,
        [
            "--mem0-dir", str(args.mem0_dir),
            "--longmemeval-file", str(args.longmemeval_file),
            "--out-jsonl", str(args.runtime_dir / "dummy_mem0.jsonl"),
            "--trace-jsonl", str(args.runtime_dir / "dummy_mem0.trace.jsonl"),
            "--openai-base-url", args.openai_base_url,
            "--llm-model", args.llm_model,
        ],
    )
    api_key = mem0_args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for mem0 CF-only run.")
    per_sample_root = args.runtime_dir / "mem0_cf_only"
    per_sample_root.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(query_records, total=len(query_records), desc="mem0 CF-only", unit="q")
    for idx, baseline_qr in enumerate(pbar, start=1):
        qid = str(baseline_qr["question_id"])
        entry = entries[qid]
        write_records = writes_by_qid.get(qid, [])
        events = mem0_bridge.collect_mem0_write_events(entry, mem0_args.preserve_session_order)
        specs = _filter_specs(build_cf_specs(
            question_type=str(baseline_qr.get("question_type") or entry.get("question_type") or "unknown"),
            query_record=baseline_qr,
            write_records=write_records,
            answer_session_ids=entry.get("answer_session_ids", []),
            max_writes=args.cf_max_writes,
            scope=args.cf_target_scope,
        ), args.cf_rule_mode)
        # mem0 is intentionally rollback-only for the first CF pass.
        specs = [spec for spec in specs if getattr(spec, "rule_id", "") == "rollback_skip"]
        cf_results = []
        for spec in specs:
            mutated_events, target_timestamp = mem0_bridge.apply_mem0_cf_spec(events, spec)
            outcome = mem0_bridge.run_mem0_replay(
                entry=entry,
                events=mutated_events,
                args=mem0_args,
                sample_storage_root=per_sample_root,
                sample_tag=f"{idx:03d}_{qid}_{spec.rule_id}_{spec.target_write_id[:8]}",
            )
            cf_results.append({
                "spec": spec,
                "cf_answer": outcome["hypothesis"],
                "cf_retrieved_write_ids": outcome["query_record"].get("retrieved_write_ids", []),
                "cf_prompt_write_ids": outcome["query_record"].get("prompt_write_ids", []),
                "target_timestamp": target_timestamp,
                "cf_extra": {
                    "n_search_results": outcome["trace"].get("n_search_results"),
                    "n_add_results": outcome["trace"].get("n_add_results"),
                },
            })
        run_records, query_summary = summarize_replay_cf(
            agent="mem0",
            entry=entry,
            baseline_query_record=baseline_qr,
            write_records=write_records,
            cf_results=cf_results,
            dominance_threshold=args.cf_dominance_threshold,
        )
        append_cf_outputs(run_path=run_path, query_path=query_path, run_records=run_records, query_summary=query_summary)


def run_ldagent(args: argparse.Namespace, entries: Dict[str, Dict], query_records: List[Dict], writes_by_qid: Dict[str, List[Dict]], run_path: Path, query_path: Path) -> None:
    import ldagent_longmemeval_bridge.run_infer as ld

    ld_args_ns = _parse_module_args(
        ld.parse_args,
        [
            "--ld-agent-dir", str(args.ld_agent_dir),
            "--longmemeval-file", str(args.longmemeval_file),
            "--out-jsonl", str(args.runtime_dir / "dummy_ld.jsonl"),
            "--trace-jsonl", str(args.runtime_dir / "dummy_ld.trace.jsonl"),
            "--openai-base-url", args.openai_base_url,
            "--llm-model", args.llm_model,
            "--session-gap-seconds", "600",
            "--dist-thres", "0.5527",
            "--no-force-flush-before-answer",
        ],
    )
    api_key = ld_args_ns.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for LD-Agent CF-only run.")
    EventMemory, Personas, Generator = ld.import_ldagent_modules(ld_args_ns.ld_agent_dir)
    logger = ld.build_logger(ld_args_ns.verbose)
    runtime_ld_args = ld.make_ld_args(ld_args_ns)
    llm_client = ld.OpenAIEmployClient(
        api_key=api_key,
        model=ld_args_ns.llm_model,
        base_url=ld_args_ns.openai_base_url,
        temperature=ld_args_ns.temperature,
        max_tokens=ld_args_ns.max_tokens,
        timeout=ld_args_ns.timeout,
    )
    pbar = tqdm(query_records, total=len(query_records), desc="LD-Agent CF-only", unit="q")
    for baseline_qr in pbar:
        qid = str(baseline_qr["question_id"])
        entry = entries[qid]
        write_records = writes_by_qid.get(qid, [])
        events = ld.collect_ldagent_write_events(entry, ld_args_ns.preserve_session_order)
        specs = _filter_specs(build_cf_specs(
            question_type=str(baseline_qr.get("question_type") or entry.get("question_type") or "unknown"),
            query_record=baseline_qr,
            write_records=write_records,
            answer_session_ids=entry.get("answer_session_ids", []),
            max_writes=args.cf_max_writes,
            scope=args.cf_target_scope,
        ), args.cf_rule_mode)
        cf_results = []
        for spec in specs:
            mutated_events, target_timestamp = ld.apply_ldagent_cf_spec(events, spec)
            outcome = ld.run_ldagent_replay(
                entry=entry,
                events=mutated_events,
                args=ld_args_ns,
                EventMemory=EventMemory,
                Personas=Personas,
                Generator=Generator,
                llm_client=llm_client,
                logger=logger,
                ld_args=runtime_ld_args,
            )
            cf_results.append({
                "spec": spec,
                "cf_answer": outcome["hypothesis"],
                "cf_retrieved_write_ids": outcome["query_record"].get("retrieved_write_ids", []),
                "cf_prompt_write_ids": outcome["query_record"].get("prompt_write_ids", []),
                "target_timestamp": target_timestamp,
            })
        run_records, query_summary = summarize_replay_cf(
            agent="ldagent",
            entry=entry,
            baseline_query_record=baseline_qr,
            write_records=write_records,
            cf_results=cf_results,
            dominance_threshold=args.cf_dominance_threshold,
        )
        append_cf_outputs(run_path=run_path, query_path=query_path, run_records=run_records, query_summary=query_summary)


def run_anna(args: argparse.Namespace, entries: Dict[str, Dict], query_records: List[Dict], writes_by_qid: Dict[str, List[Dict]], run_path: Path, query_path: Path) -> None:
    import anna_longmemeval_bridge.run_infer as anna

    anna_args = _parse_module_args(
        anna.parse_args,
        [
            "--anna-agent-dir", str(args.anna_agent_dir),
            "--longmemeval-file", str(args.longmemeval_file),
            "--out-jsonl", str(args.runtime_dir / "dummy_anna.jsonl"),
            "--trace-jsonl", str(args.runtime_dir / "dummy_anna.trace.jsonl"),
            "--openai-base-url", args.openai_base_url,
            "--llm-model", args.llm_model,
            "--disable-full-tertiary-init",
            "--disable-need-check",
        ],
    )
    api_key = anna_args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for Anna CF-only run.")
    llm = anna.OpenAITextClient(
        api_key=api_key,
        model=anna_args.llm_model,
        base_url=anna_args.openai_base_url,
        temperature=anna_args.temperature,
        max_tokens=anna_args.max_tokens,
        timeout=anna_args.timeout,
    )
    retriever = anna.AnnaRetriever(
        anna_agent_dir=anna_args.anna_agent_dir,
        api_key=api_key,
        model=anna_args.llm_model,
        base_url=anna_args.openai_base_url,
    )
    pbar = tqdm(query_records, total=len(query_records), desc="Anna CF-only", unit="q")
    for baseline_qr in pbar:
        qid = str(baseline_qr["question_id"])
        entry = entries[qid]
        write_records = writes_by_qid.get(qid, [])
        units = anna.collect_anna_write_events(entry, anna_args.preserve_session_order)
        specs = _filter_specs(build_cf_specs(
            question_type=str(baseline_qr.get("question_type") or entry.get("question_type") or "unknown"),
            query_record=baseline_qr,
            write_records=write_records,
            answer_session_ids=entry.get("answer_session_ids", []),
            max_writes=args.cf_max_writes,
            scope=args.cf_target_scope,
        ), args.cf_rule_mode)
        cf_results = []
        for spec in specs:
            mutated_units, target_timestamp = anna.apply_anna_cf_spec(qid, units, spec)
            outcome = anna.run_anna_replay(
                entry=entry,
                units=mutated_units,
                args=anna_args,
                retriever=retriever,
                llm=llm,
            )
            cf_results.append({
                "spec": spec,
                "cf_answer": outcome["hypothesis"],
                "cf_retrieved_write_ids": outcome["query_record"].get("retrieved_write_ids", []),
                "cf_prompt_write_ids": outcome["query_record"].get("prompt_write_ids", []),
                "target_timestamp": target_timestamp,
            })
        run_records, query_summary = summarize_replay_cf(
            agent="anna",
            entry=entry,
            baseline_query_record=baseline_qr,
            write_records=write_records,
            cf_results=cf_results,
            dominance_threshold=args.cf_dominance_threshold,
        )
        append_cf_outputs(run_path=run_path, query_path=query_path, run_records=run_records, query_summary=query_summary)


def run_theanine(args: argparse.Namespace, entries: Dict[str, Dict], query_records: List[Dict], writes_by_qid: Dict[str, List[Dict]], run_path: Path, query_path: Path) -> None:
    import theanine_longmemeval_bridge.run_infer as theanine

    theanine_args = _parse_module_args(
        theanine.parse_args,
        [
            "--theanine-dir", str(args.theanine_dir),
            "--longmemeval-file", str(args.longmemeval_file),
            "--out-jsonl", str(args.runtime_dir / "dummy_theanine.jsonl"),
            "--trace-jsonl", str(args.runtime_dir / "dummy_theanine.trace.jsonl"),
            "--llm-model", args.llm_model,
            "--runtime-dir", str(args.runtime_dir / "theanine_cf_only"),
        ],
    )
    pbar = tqdm(query_records, total=len(query_records), desc="THEANINE CF-only", unit="q")
    for idx, baseline_qr in enumerate(pbar, start=1):
        qid = str(baseline_qr["question_id"])
        entry = entries[qid]
        write_records = writes_by_qid.get(qid, [])
        events = theanine.collect_theanine_write_events(entry, theanine_args.history_sessions, theanine_args.preserve_session_order)
        specs = _filter_specs(build_cf_specs(
            question_type=str(baseline_qr.get("question_type") or entry.get("question_type") or "unknown"),
            query_record=baseline_qr,
            write_records=write_records,
            answer_session_ids=entry.get("answer_session_ids", []),
            max_writes=args.cf_max_writes,
            scope=args.cf_target_scope,
        ), args.cf_rule_mode)
        cf_results = []
        for spec in specs:
            mutated_events, target_timestamp = theanine.apply_theanine_cf_spec(events, spec)
            hypothesis, trace = theanine.run_theanine_for_entry(
                entry=entry,
                theanine_dir=theanine_args.theanine_dir,
                runtime_dir=theanine_args.runtime_dir / "cf_runs",
                llm_model=theanine_args.llm_model,
                temperature=theanine_args.temperature,
                history_sessions=theanine_args.history_sessions,
                preserve_session_order=theanine_args.preserve_session_order,
                omit_question_date=theanine_args.omit_question_date,
                dry_run=False,
                verbose_upstream=theanine_args.verbose_upstream,
                seed=theanine_args.seed + idx,
                events=mutated_events,
            )
            _, outcome_qr = theanine.build_theanine_replay_audit(entry, mutated_events, trace, hypothesis)
            cf_results.append({
                "spec": spec,
                "cf_answer": hypothesis,
                "cf_retrieved_write_ids": outcome_qr.get("retrieved_write_ids", []),
                "cf_prompt_write_ids": outcome_qr.get("prompt_write_ids", []),
                "target_timestamp": target_timestamp,
            })
        run_records, query_summary = summarize_replay_cf(
            agent="theanine",
            entry=entry,
            baseline_query_record=baseline_qr,
            write_records=write_records,
            cf_results=cf_results,
            dominance_threshold=args.cf_dominance_threshold,
        )
        append_cf_outputs(run_path=run_path, query_path=query_path, run_records=run_records, query_summary=query_summary)


def main() -> None:
    args = parse_args()
    args.runtime_dir.mkdir(parents=True, exist_ok=True)

    audit_query_path, audit_write_path = _derive_audit_paths(args.baseline_trace_jsonl)
    if args.baseline_audit_queries is not None:
        audit_query_path = args.baseline_audit_queries
    if args.baseline_audit_writes is not None:
        audit_write_path = args.baseline_audit_writes

    run_path, query_path = _derive_cf_only_paths(args.baseline_trace_jsonl, args.cf_tag)
    _prepare_output(run_path)
    _prepare_output(query_path)

    entries = _load_dataset(args.longmemeval_file)
    query_records = _iter_baseline_queries(audit_query_path)
    write_records = _load_jsonl(audit_write_path)
    writes_by_qid = _group_writes_by_qid(write_records)

    query_records = [qr for qr in query_records if str(qr.get("question_id")) in entries]
    if not query_records:
        raise RuntimeError(f"No baseline audit queries found in {audit_query_path}")

    print(f"Loaded {len(query_records)} baseline queries from {audit_query_path}")
    print(f"Writing CF-only outputs to {run_path} and {query_path}")

    if args.agent == "share":
        run_share(args, entries, query_records, writes_by_qid, run_path, query_path)
    elif args.agent == "memoryos":
        run_memoryos(args, entries, query_records, writes_by_qid, run_path, query_path)
    elif args.agent == "mem0":
        run_mem0(args, entries, query_records, writes_by_qid, run_path, query_path)
    elif args.agent == "ldagent":
        run_ldagent(args, entries, query_records, writes_by_qid, run_path, query_path)
    elif args.agent == "anna":
        run_anna(args, entries, query_records, writes_by_qid, run_path, query_path)
    elif args.agent == "theanine":
        run_theanine(args, entries, query_records, writes_by_qid, run_path, query_path)
    else:
        raise AssertionError(args.agent)


if __name__ == "__main__":
    main()
