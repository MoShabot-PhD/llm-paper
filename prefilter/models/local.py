import json
import re

import pandas as pd
from ollama import chat

from prefilter.config import LOCAL_OPTIONS
from prefilter.prompts import PROMPT


class LocalModels:
    def __init__(self, model: str, yt_path: str):
        self.model = model 
        self.yt_path = yt_path

    def extract_json_object(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"```(?:json)?|```", "", text).strip()

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        start = text.find("{")
        if start == -1:
            return ""
        text = text[start:]

        brace_count = 0
        in_string = False
        escape = False

        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                    continue

            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[: i + 1]

        return ""

    def run_annotations_kp(self, output_path: str, title_col="title", desc_col = "description"):
        df = pd.read_csv(self.yt_path, sep=None, engine="python", on_bad_lines="skip")
        
        result_decisions = []
        result_confidence = []
        result_reasons = []
        prompt = PROMPT
        partial_path = f"{output_path}.partial.csv"
        start_idx = 0

        try:
            partial_df = pd.read_csv(partial_path, sep=None, engine="python", on_bad_lines="skip")
            required_cols = {"DECISIONS_AI", "CONFIDENCE_AI", "REASONS_AI"}
            if required_cols.issubset(set(partial_df.columns)):
                rows_done = min(len(partial_df), len(df))
                result_decisions = partial_df["DECISIONS_AI"].tolist()[:rows_done]
                result_confidence = partial_df["CONFIDENCE_AI"].tolist()[:rows_done]
                result_reasons = partial_df["REASONS_AI"].tolist()[:rows_done]
                start_idx = rows_done
                if start_idx > 0:
                    print(f"Resuming from checkpoint: {start_idx}/{len(df)} rows already processed.", flush=True)
        except FileNotFoundError:
            pass

        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            title = "" if row.get(title_col) is None else str(row.get(title_col)).strip()
            description = "" if row.get(desc_col) is None else str(row.get(desc_col)).strip()

            response = chat(
                model=self.model,
                format="json",
                options=LOCAL_OPTIONS,
                messages=[
                    {
                        "role": "system",
                        "content": f"{prompt}",
                    },
                    {"role": "user", "content": (
                        f"Now evaluate the following video:\n"
                        f"Video metadata:\n"
                        f"Title: {title}\n"
                        f"Description: {description}"
                    ).strip()}
                ],
            )
            
            output_text = response["message"]["content"]
            
            if not output_text or output_text.strip() == "":
                output_text = response["message"].get("thinking", "")
            
            output_text = re.sub(r"```json|```", "", output_text).strip()
            
            json_start = output_text.find('{')
            if json_start != -1:
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(output_text)):
                    if output_text[i] == '{':
                        brace_count += 1
                    elif output_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end != -1:
                    output_text = output_text[json_start:json_end]
                    
                
            try:
                json_text = json.loads(self.extract_json_object(output_text))
                ai_decision = json_text.get('decision', 'DISCARD')
                ai_confidence = float(json_text.get('confidence', 0.0))
                ai_reason = json_text.get('reason', 'Failed to parse response')
            except json.JSONDecodeError as e:
                print(f"JSON decode error for video '{title}': {e}")
                print(f"Problematic output_text: '{output_text[:500]}'")
                ai_decision = 'DISCARD'
                ai_confidence = 0.0
                ai_reason = f'JSON parsing failed: {str(e)}'
            except (TypeError, ValueError):
                ai_decision = 'DISCARD'
                ai_confidence = 0.0
                ai_reason = 'Invalid confidence value'
            
            result_decisions.append(ai_decision)
            result_confidence.append(ai_confidence)
            result_reasons.append(ai_reason)
            
            print(f"[{idx+1}/{len(df)}] Video '{title}': {ai_decision} (conf: {ai_confidence})", flush=True)

            if (idx + 1) % 5 == 0:
                tmp_df = df.iloc[: idx + 1].copy()
                tmp_df["DECISIONS_AI"] = result_decisions
                tmp_df["CONFIDENCE_AI"] = result_confidence
                tmp_df["REASONS_AI"] = result_reasons
                tmp_df.to_csv(partial_path, index=False)
            
        
        df["DECISIONS_AI"] = result_decisions
        df["CONFIDENCE_AI"] = result_confidence
        df["REASONS_AI"] = result_reasons      
        df.to_csv(output_path, index = False)
        return df
