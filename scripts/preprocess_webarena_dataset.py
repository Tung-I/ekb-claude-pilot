#!/usr/bin/env python3
"""
Usage:
python scripts/preprocess_webarena_dataset.py \
  --input /home/tungichen_umass_edu/ekb-claude-pilot/data/webarena/test.raw.json \
  --output-dir /home/tungichen_umass_edu/ekb-claude-pilot/data/webarena
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PLACEHOLDER_ENV_MAP = {
    "__SHOPPING__": "SHOPPING",
    "__SHOPPING_ADMIN__": "SHOPPING_ADMIN",
    "__REDDIT__": "REDDIT",
    "__GITLAB__": "GITLAB",
    "__MAP__": "MAP",
    "__WIKIPEDIA__": "WIKIPEDIA",
    "__HOMEPAGE__": "HOMEPAGE",
}

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n

def normalize_to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def placeholder_env_value(env_name: str) -> Optional[str]:
    value = os.environ.get(env_name)
    if not value:
        return None
    return value.rstrip("/")

def render_placeholders(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = text
    for placeholder, env_name in PLACEHOLDER_ENV_MAP.items():
        value = placeholder_env_value(env_name)
        if value:
            out = out.replace(placeholder, value)
    return out

def maybe_render_start_urls(start_url: Any) -> List[str]:
    urls = normalize_to_list(start_url)
    out: List[str] = []
    for u in urls:
        if u is None:
            continue
        parts = [p.strip() for p in re.split(r"\s*\|AND\|\s*", str(u)) if p.strip()]
        out.extend(render_placeholders(p) for p in parts)
    return out

def choose_reference_answer(eval_obj: Dict[str, Any]) -> Optional[str]:
    if not isinstance(eval_obj, dict):
        return None
    ref_answers = eval_obj.get("reference_answers")
    if isinstance(ref_answers, dict):
        if "exact_match" in ref_answers and ref_answers["exact_match"] not in (None, "", "N/A"):
            return str(ref_answers["exact_match"])
        for v in ref_answers.values():
            if isinstance(v, (str, int, float)) and str(v).strip() not in ("", "N/A"):
                return str(v)
    return None

def build_question(intent: str, start_urls: List[str], sites: List[str]) -> str:
    lines: List[str] = []
    if start_urls:
        if len(start_urls) == 1:
            lines.append(f"Start URL: {start_urls[0]}")
        else:
            lines.append("Start URLs:")
            lines.extend(f"- {u}" for u in start_urls)
    if sites:
        lines.append("Sites: " + ", ".join(str(s) for s in sites))
    lines.append("Task:")
    lines.append(intent.strip())
    return "\n".join(lines).strip()

def preprocess_row(row: Dict[str, Any], benchmark_name: str, split_name: str, source_path: str) -> Dict[str, Any]:
    task_id = row.get("task_id")
    intent = str(row.get("intent", "")).strip()
    sites = [str(s) for s in normalize_to_list(row.get("sites")) if s is not None]
    start_urls = maybe_render_start_urls(row.get("start_url") or row.get("start_urls"))
    final_answer = choose_reference_answer(row.get("eval", {}))
    return {
        "task_id": str(task_id),
        "question": build_question(intent, start_urls, sites),
        "final_answer": final_answer,
        "benchmark": benchmark_name,
        "split": split_name,
        "level": None,
        "sites": sites,
        "start_urls": start_urls,
        "intent": intent,
        "intent_template": row.get("intent_template"),
        "intent_template_id": row.get("intent_template_id"),
        "instantiation_dict": row.get("instantiation_dict", {}),
        "require_login": row.get("require_login"),
        "storage_state": row.get("storage_state"),
        "geolocation": row.get("geolocation"),
        "require_reset": row.get("require_reset"),
        "reference_url": row.get("eval", {}).get("reference_url") if isinstance(row.get("eval"), dict) else None,
        "raw_eval": row.get("eval"),
        "source_json": source_path,
    }

def infer_benchmark_name(input_path: Path) -> str:
    return "webarena-verified" if "verified" in input_path.name.casefold() else "webarena"

def infer_split_name(input_path: Path) -> str:
    name = input_path.name.casefold()
    if "raw" in name or "test" in name: return "test"
    if "hard" in name: return "hard"
    if "full" in name or "verified" in name: return "full"
    return "unknown"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/webarena")
    parser.add_argument("--benchmark-name", type=str, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_json(input_path)
    if isinstance(raw, dict):
        rows = raw.get("tasks") or raw.get("data")
        if not isinstance(rows, list):
            raise ValueError(f"Unsupported JSON object structure in {input_path}")
    elif isinstance(raw, list):
        rows = raw
    else:
        raise ValueError(f"Unsupported JSON top-level type in {input_path}: {type(raw)}")
    benchmark_name = args.benchmark_name or infer_benchmark_name(input_path)
    split_name = args.split_name or infer_split_name(input_path)
    processed = [preprocess_row(r, benchmark_name, split_name, str(input_path))
                 for r in rows
                 if isinstance(r, dict) and r.get("task_id") is not None and str(r.get("intent", "")).strip()]
    out_jsonl = output_dir / "webarena_prepared.jsonl"
    n = write_jsonl(out_jsonl, processed)
    summary = {
        "input": str(input_path),
        "output_jsonl": str(out_jsonl),
        "num_rows_written": n,
        "benchmark": benchmark_name,
        "split": split_name,
        "num_with_reference_answer": sum(1 for r in processed if r.get("final_answer") not in (None, "")),
        "sites_observed": sorted({s for r in processed for s in r.get("sites", [])}),
        "note": "Prepared for intent-centric Claude trace collection. WebArena tasks are browser-interaction tasks; this JSONL is not an official evaluator input."
    }
    write_json(output_dir / "webarena_prepared_summary.json", summary)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
