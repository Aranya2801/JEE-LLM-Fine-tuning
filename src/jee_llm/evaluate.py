"""Evaluate a model (base, or base + LoRA adapter) on JEEBench or any Problem jsonl.

    # greedy
    python -m jee_llm.evaluate --model Qwen/Qwen2.5-7B-Instruct --data dataset/jeebench.jsonl --out results/base

    # 8 samples: gives pass@k, maj@8 and the abstention curve
    python -m jee_llm.evaluate --model Qwen/Qwen2.5-7B-Instruct --adapter checkpoints/grpo/final \\
        --data dataset/jeebench.jsonl --num-samples 8 --temperature 0.7 --out results/grpo

    # re-score saved completions without generating again (e.g. after fixing the answer parser)
    python -m jee_llm.evaluate --rescore results/grpo/completions.jsonl --out results/grpo_rescored

Outputs in --out: completions.jsonl (everything, for error analysis),
metrics.json and report.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from .data import load_problems, read_jsonl, write_jsonl
from .metrics import ItemResult, abstention_curve, summarize
from .prompts import build_messages

log = logging.getLogger("jee_llm.eval")


def _pct(x) -> str:
    return f"{100 * x:.1f}" if isinstance(x, float) else str(x)


def render_report(meta: dict, metrics: dict, curve: list[dict]) -> str:
    lines = [f"# Eval: {meta.get('model', 'rescored')}", ""]
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    cols = [c for c in ("n", "accuracy@1", "pass@1", "pass@4", "pass@8", "maj@8", "jee_score_pct", "no_answer_rate")
            if c in metrics["overall"]]
    for title, section in (("By subject", metrics["by_subject"]), ("By question type", metrics["by_type"])):
        lines += [f"## {title}", "", "| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for name, block in list(section.items()) + [("**overall**", metrics["overall"])]:
            lines.append(f"| {name} | " + " | ".join(_pct(block.get(c, "")) for c in cols) + " |")
        lines.append("")
    lines.append("Percentages, except `n`. `jee_score_pct` is marks obtained / max marks with negative marking.")

    if metrics["overall"].get("samples_per_item", 1) > 1:
        lines += ["", "## Abstention (answer only if majority-vote share >= tau)", "",
                  "| tau | score | max | score % | attempted |", "|---|---|---|---|---|"]
        for row in curve:
            lines.append(f"| {row['threshold']:.2f} | {row['score']:.0f} | {row['max']:.0f} | "
                         f"{100 * row['score_pct']:.1f} | {row['attempted']} |")
    return "\n".join(lines) + "\n"


def write_outputs(out: Path, results: list[ItemResult], meta: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "completions.jsonl", (
        {"id": r.id, "subject": r.subject, "question_type": r.question_type, "gold": r.gold,
         "completions": r.completions, "preds": r.preds, "correct": r.correct,
         "vote": r.vote, "vote_share": r.vote_share}
        for r in results
    ))
    metrics = summarize(results)
    curve = abstention_curve(results)
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "metrics": metrics, "abstention": curve}, indent=2))
    (out / "report.md").write_text(render_report(meta, metrics, curve))
    return metrics


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (output of a training stage)")
    ap.add_argument("--data", help="Problem jsonl, e.g. dataset/jeebench.jsonl")
    ap.add_argument("--rescore", help="completions.jsonl from an earlier run; skips generation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--subjects", nargs="*", default=None, help="e.g. math physics")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-skip", action="store_true", help="tell the model it may answer SKIP on MCQs")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.rescore:
        results = [ItemResult.build(r["id"], r["subject"], r["question_type"], r["gold"], r["completions"])
                   for r in read_jsonl(args.rescore)]
        meta = {"rescored_from": args.rescore}
    else:
        if not (args.model and args.data):
            ap.error("--model and --data are required unless --rescore is given")
        problems = load_problems(args.data)
        if args.subjects:
            problems = [p for p in problems if p.subject in args.subjects]
        problems = problems[: args.limit] if args.limit else problems
        log.info("%d problems", len(problems))

        from .generation import Generator

        gen = Generator(args.model, adapter=args.adapter, backend=args.backend)
        t0 = time.time()
        completions = gen.generate(
            [build_messages(p, allow_skip=args.allow_skip) for p in problems],
            n=args.num_samples, temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_new_tokens, seed=args.seed,
        )
        elapsed = time.time() - t0
        results = [ItemResult.build(p.id, p.subject, p.question_type, p.answer, c)
                   for p, c in zip(problems, completions, strict=True)]
        meta = {
            "model": args.model, "adapter": args.adapter, "data": args.data, "n_problems": len(problems),
            "num_samples": args.num_samples, "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens, "allow_skip": args.allow_skip,
            "backend": gen.backend, "seconds": round(elapsed, 1),
        }

    metrics = write_outputs(Path(args.out), results, meta)
    o = metrics["overall"]
    log.info("accuracy@1 %.3f | jee score %.0f/%.0f (%.1f%%) -> %s",
             o["accuracy@1"], o["jee_score"], o["jee_max"], 100 * o["jee_score_pct"], args.out)
    return metrics


if __name__ == "__main__":
    main()
