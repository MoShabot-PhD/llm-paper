import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from prefilter.config import API_MAX_TOKENS, API_TEMPERATURE
from prefilter.prompts import PROMPT


class OpenRouterModel:
    BATCH_SUBMIT_URL = "https://openrouter.ai/api/beta/batches"

    def __init__(self, api_key: str, yt_path: str, max_rows: int | None = None):
        self.api_key = api_key
        self.yt_path = yt_path
        self.max_rows = max_rows

    def safe_str(self, x) -> str:
        if x is None:
            return ""
        return str(x).strip()

    def sanitize_batch_custom_id(self, raw_id: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id).strip("_")
        if 1 <= len(cleaned) <= 64:
            return cleaned
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
        return digest[:64]

    def _parse_batch_content(self, content: str):
        content = (content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    _PREFLIGHT_FLAG_RE = re.compile(r"Preflight failed for batch request '([^']+)'")

    def _classify_batch_item(self, response: dict | None) -> tuple[dict | None, str]:
        if not response:
            return None, "empty_response"
        body = response.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            return None, "empty_response"
        choice = choices[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason")
        if message.get("refusal"):
            return None, f"model_refusal: {str(message['refusal'])[:160]}"
        if finish_reason == "content_filter":
            return None, "content_filter"
        content = message.get("content")
        if not content:
            return None, "empty_response"
        parsed = self._parse_batch_content(content)
        if parsed is None:
            return None, "unparseable_json"
        return parsed, "ok"

    def _submit_and_poll_batch(
        self,
        chunk: list[tuple[str, str, str]],
        model: str,
        system_prompt: str,
        headers: dict,
        poll_interval: int,
        timeout_seconds: int,
        max_moderation_retries: int = 10,
    ) -> tuple[dict[str, dict | None], dict[str, str], list[dict]]:
        remaining = list(chunk)
        excluded: dict[str, str] = {}
        usage_records: list[dict] = []

        for attempt in range(max_moderation_retries + 1):
            if not remaining:
                break

            sanitized_to_raw: dict[str, str] = {}
            batch_requests = []
            for cid, title, desc in remaining:
                user_content = (
                    f"Now evaluate the following video:\n"
                    f"Video metadata:\n"
                    f"Title: {title}\n"
                    f"Description: {desc}"
                ).strip()
                sid = self.sanitize_batch_custom_id(cid)
                sanitized_to_raw[sid] = cid
                batch_requests.append({
                    "custom_id": sid,
                    "body": {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": API_TEMPERATURE,
                        "max_tokens": API_MAX_TOKENS,
                    },
                })

            submit_payload = {
                "endpoint": "/v1/chat/completions",
                "model": model,
                "requests": batch_requests,
            }
            submit_resp = requests.post(self.BATCH_SUBMIT_URL, headers=headers, json=submit_payload, timeout=120)
            submit_resp.raise_for_status()
            batch = submit_resp.json()
            batch_id = batch.get("id")
            print(f"  Batch submitted: {batch_id} ({len(remaining)} items), status={batch.get('status')}")

            start = time.time()
            status = batch.get("status")
            data = batch
            while status not in ("completed", "failed", "expired", "cancelled"):
                if time.time() - start > timeout_seconds:
                    raise TimeoutError(f"Batch {batch_id} did not finish within {timeout_seconds}s")
                time.sleep(poll_interval)
                status_resp = requests.get(f"{self.BATCH_SUBMIT_URL}/{batch_id}", headers=headers, timeout=60)
                status_resp.raise_for_status()
                data = status_resp.json()
                status = data.get("status")
                print(f"  [{batch_id}] status={status} counts={data.get('request_counts')}")

            if status == "completed":
                usage = data.get("usage") or {}
                usage_records.append({
                    "batch_id": batch_id,
                    "model": model,
                    "n_requests": len(remaining),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost_usd": usage.get("cost", 0.0),
                    "request_counts": data.get("request_counts"),
                    "resolved_models": sorted({
                        ((item.get("response") or {}).get("body") or {}).get("model")
                        for item in data.get("results", [])
                    } - {None}),
                })

                out: dict[str, dict | None] = {raw_id: None for raw_id in excluded}
                statuses: dict[str, str] = {raw_id: "excluded_preflight" for raw_id in excluded}
                for item in data.get("results", []):
                    sid = item.get("custom_id")
                    raw_id = sanitized_to_raw.get(sid, sid)
                    parsed, item_status = self._classify_batch_item(item.get("response"))
                    out[raw_id] = parsed
                    statuses[raw_id] = item_status
                if excluded:
                    print(f"  Excluded {len(excluded)} item(s) due to provider moderation: {list(excluded.values())}")
                n_bad = sum(1 for s in statuses.values() if s != "ok")
                if n_bad:
                    print(f"  {n_bad}/{len(statuses)} item(s) in this batch had no usable result "
                          f"(refusal/content-filter/parse-error/excluded) -- see status field in output.")
                return out, statuses, usage_records

            error_msg = (data.get("error") or {}).get("message", "")
            m = self._PREFLIGHT_FLAG_RE.search(error_msg)
            if status == "failed" and m:
                flagged_sid = m.group(1)
                flagged_raw = sanitized_to_raw.get(flagged_sid, flagged_sid)
                excluded[flagged_raw] = error_msg
                print(f"  Moderation flag on {flagged_raw!r}, excluding and retrying chunk: {error_msg}")
                remaining = [r for r in remaining if r[0] != flagged_raw]
                continue

            raise RuntimeError(f"Batch {batch_id} ended with status={status!r}: {json.dumps(data)[:2000]}")

        statuses = {raw_id: "excluded_preflight" for raw_id in excluded}
        return {raw_id: None for raw_id in excluded}, statuses, usage_records

    def run_batch_classifier(
        self,
        platform: str,
        model: str,
        batch_dir: str,
        title_col: str = "title",
        desc_col: str = "description",
        poll_interval: int = 30,
        timeout_seconds: int = 24 * 3600,
        chunk_size: int = 100,
    ) -> str:
        batch_dir = Path(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.yt_path, sep=None, engine="python", on_bad_lines="skip")

        if self.max_rows is not None:
            df = df.head(self.max_rows)

        seen_ids: set[str] = set()
        rows: list[tuple[str, str, str]] = []
        for _, row in df.iterrows():
            cid = self.safe_str(row.get("url"))
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            rows.append((cid, self.safe_str(row.get(title_col)), self.safe_str(row.get(desc_col))))

        model_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
        out_path = batch_dir / f"{model_slug}_{platform}_batch_results.jsonl"
        usage_path = batch_dir / f"{model_slug}_{platform}_usage.jsonl"
        summary_path = batch_dir / f"{model_slug}_{platform}_usage_summary.json"

        done_ids: set[str] = set()
        if out_path.exists():
            with out_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        done_ids.add(json.loads(line)["custom_id"])
            print(f"Resuming {out_path}: {len(done_ids)} rows already done.")

        remaining_rows = [r for r in rows if r[0] not in done_ids]
        print(f"Classifying {len(remaining_rows)}/{len(rows)} remaining rows for platform={platform!r} "
              f"with model={model!r} in chunks of {chunk_size} ...")

        if remaining_rows:
            system_prompt = PROMPT
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            chunks = [remaining_rows[i:i + chunk_size] for i in range(0, len(remaining_rows), chunk_size)]
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i + 1}/{len(chunks)} ({len(chunk)} items) ...")
                chunk_out, chunk_statuses, chunk_usage = self._submit_and_poll_batch(
                    chunk, model, system_prompt, headers, poll_interval, timeout_seconds
                )
                with out_path.open("a", encoding="utf-8") as f:
                    for cid, _, _ in chunk:
                        f.write(json.dumps({
                            "batch_id": model_slug,
                            "custom_id": cid,
                            "result": chunk_out.get(cid),
                            "status": chunk_statuses.get(cid, "empty_response"),
                        }, ensure_ascii=False) + "\n")
                with usage_path.open("a", encoding="utf-8") as f:
                    for rec in chunk_usage:
                        f.write(json.dumps({**rec, "chunk_index": i}, ensure_ascii=False) + "\n")
                print(f"Chunk {i + 1}/{len(chunks)} written to {out_path}.")

        n_ok = 0
        status_counts: dict[str, int] = {}
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                status = row.get("status") or ("ok" if row.get("result") is not None else "empty_response")
                status_counts[status] = status_counts.get(status, 0) + 1
                if row.get("result") is not None:
                    n_ok += 1
        print(f"All chunks done: {n_ok}/{len(rows)} parsed successfully. Results written to: {out_path}")

        totals = {"n_batches": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
        if usage_path.exists():
            with usage_path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    totals["n_batches"] += 1
                    totals["prompt_tokens"] += rec.get("prompt_tokens", 0)
                    totals["completion_tokens"] += rec.get("completion_tokens", 0)
                    totals["total_tokens"] += rec.get("total_tokens", 0)
                    totals["cost_usd"] += rec.get("cost_usd", 0.0)

        summary = {
            "model": model,
            "platform": platform,
            "n_rows": len(rows),
            "status_counts": status_counts,
            **totals,
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        refused_or_bad = {k: v for k, v in status_counts.items() if k != "ok"}
        print(f"Tokens: prompt={totals['prompt_tokens']:,} completion={totals['completion_tokens']:,} "
              f"total={totals['total_tokens']:,}  cost (billed by OpenRouter): ${totals['cost_usd']:.4f}")
        if refused_or_bad:
            print(f"Non-ok statuses: {refused_or_bad}")
        print(f"Usage log: {usage_path}")
        print(f"Summary: {summary_path}")
        return str(out_path)
