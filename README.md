# LLM metadata pre-filter

Code for scraping YouTube metadata and pre-screening videos with LLMs (KEEP/DISCARD from title and description) before download.

Models: GPT-5.2 and Claude Sonnet 4.6 (OpenRouter Batch API), and Qwen3.6 and Gemma4:31B run locally through Ollama. All models receive the same system prompt and the same input format.

## Setup

Requires Python 3.11.

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OpenRouter API key
```

For the local models, install [Ollama](https://ollama.com) and pull the models:

```bash
ollama pull qwen3.6
ollama pull gemma4:31b
```

The scripts can be run from any directory. Inputs are read from `data/` and outputs are written to `results/`. Neither folder is part of the repository.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Scrape metadata with yt-dlp (no download). Search queries are emotion × subject × setting, built from `keywords.csv`, keeping the top 10 results per query. | `python -m prefilter.youtube_scraper` | `data/yt_scraped_exp.csv` |
| 2. Annotate manually (KEEP/DISCARD). | none | `data/triaged_videos.xlsx` |
| 3. Scrape more videos on queries not used yet. | `python -m prefilter.scrape_expansion` | `data/youtube_videos_expansion.csv` |
| 4. Merge new and existing videos into one file for annotation. | `python -m prefilter.merge_expansion` | `data/triaged_videos_expanded.xlsx` |
| 5. Annotate all videos with GPT-5.2 and Claude. | `python -m prefilter.annotate_api` | `results/annotations/` |
| 6. Annotate all videos with Qwen3.6 and Gemma4:31B. | `python -m prefilter.annotate_local` | `results/annotations/` |

Steps 5 and 6 accept `--smoke-test` to run on 3 videos first. Both steps resume where they stopped if interrupted.

The system prompt is in [`prefilter/prompts.py`](prefilter/prompts.py). Each model returns JSON with `decision`, `confidence`, `emotion_likelihood`, `ai_generated_likelihood` and `reason`.

## Reproducibility

All settings are in [`prefilter/config.py`](prefilter/config.py): paths, seed, number of results per query, model identifiers, temperatures, token limit and batch size.

**Input data.** The annotation steps only need `data/triaged_videos_expanded.xlsx` with one row per video and at least the columns `url`, `title` and `description`. The other columns produced by the scraper are `query`, `platform`, `duration`, `view_count`, `uploader` and `upload_date`. Manual labels are stored in columns starting with `RESULT METADATA` and `RESULT VIDEO`, with an annotator suffix (for example `RESULT METADATA_JA`).

**Scraping.** Query order is shuffled with a fixed seed, so the same queries are generated on every run. YouTube search results change over time, however, so re-scraping does not return the same videos. To reproduce the annotations exactly, start from the scraped metadata instead of re-scraping.

**Local models.** The exact model versions are pinned by their Ollama digest in `config.py`. `annotate_local` stops if an installed model does not match. Generation uses temperature 0.3, top_p 0.9 and a fixed seed.

| Model | Ollama tag | Parameters | Quantization | Digest (first 12 characters) |
|---|---|---|---|---|
| Qwen3.6 | `qwen3.6:latest` | 36.0B | Q4_K_M | `07d35212591f` |
| Gemma4 | `gemma4:31b` | 31.3B | Q4_K_M | `6316f0629137` |

**API models.** These run with temperature 1, so repeated runs give different outputs. The model version that OpenRouter resolved for each batch is recorded as `resolved_models` in `results/annotations/*_usage.jsonl`, together with token counts and cost.

**Run manifests.** Each annotation run writes `manifest_<model>.json` next to its output. It records the model, generation parameters, the SHA-256 of the prompt and of the input file, the start time, the Python and package versions, and the platform.
