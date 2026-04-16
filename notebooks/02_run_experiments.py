"""
02 — Run Experiments
Iterates over all 570 pending runs in SQLite, calls each API,
saves responses. Fully resumable — safe to interrupt and restart.
Run: python3 notebooks/04_run_experiments.py
Or run cells individually in VS Code Interactive.
"""

# %% ── Cell 1: Imports + setup ────────────────────────────────────────────────
import os, time, sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv(".env")

import anthropic
from openai import OpenAI
import google.genai as genai

DB_PATH = "experiment.db"

claude_client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client  = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

TEMP       = 0
MAX_TOKENS = 600
SLEEP_SEC  = 2   # between calls — respect rate limits

print("Imports OK. DB:", DB_PATH)


# %% ── Cell 2: API call functions ─────────────────────────────────────────────

def call_claude(prompt: str) -> tuple[str, str]:
    """Returns (response_text, model_version)."""
    msg = claude_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=MAX_TOKENS,
        temperature=TEMP,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text, "claude-sonnet-4-5-20250929"


def call_gpt4o(prompt: str) -> tuple[str, str]:
    resp = openai_client.chat.completions.create(
        model="gpt-5.4",
        max_completion_tokens=MAX_TOKENS,   # gpt-5.4 uses max_completion_tokens
        temperature=TEMP,
        messages=[{"role": "user", "content": prompt}]
    )
    version = resp.model  # e.g. gpt-5.4-2026-03-05
    return resp.choices[0].message.content, version


def call_gemini(prompt: str) -> tuple[str, str]:
    # thinking_budget=0: disables extended thinking for fair comparison with
    # Claude and GPT-5.4 (both run in standard non-thinking mode). Without this,
    # thinking tokens consume the 600-token budget, truncating actual responses.
    resp = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            max_output_tokens=MAX_TOKENS,
            temperature=TEMP,
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
        )
    )
    text = resp.text or resp.candidates[0].content.parts[0].text or ""
    return text, "gemini-2.5-flash"


MODEL_FNS = {
    "claude": call_claude,
    "gpt4o":  call_gpt4o,
    "gemini": call_gemini,
}

print("API functions defined.")
print(f"  temperature={TEMP}, max_tokens={MAX_TOKENS}, no system prompt")
print(f"  Models: claude-sonnet-4-5-20250929 | gpt-5.4 | gemini-2.5-flash")


# %% ── Cell 3: Fetch pending runs ─────────────────────────────────────────────

def fetch_pending(con: sqlite3.Connection, model_filter: str | None = None) -> list[dict]:
    """Return all pending runs, optionally filtered by model."""
    query = "SELECT run_id, model, full_prompt FROM runs WHERE status='pending'"
    params = []
    if model_filter:
        query += " AND model=?"
        params.append(model_filter)
    query += " ORDER BY run_id"
    rows = con.execute(query, params).fetchall()
    return [{"run_id": r[0], "model": r[1], "full_prompt": r[2]} for r in rows]


con = sqlite3.connect(DB_PATH)
pending = fetch_pending(con)
con.close()

print(f"Pending runs: {len(pending)}")

# Count by model
from collections import Counter
counts = Counter(r["model"] for r in pending)
for m, n in sorted(counts.items()):
    print(f"  {m}: {n} pending")


# %% ── Cell 4: Run all pending (batched + resumable) ─────────────────────────
"""
Design:
  - Every call is a FRESH, INDEPENDENT API request — no conversation history,
    no session state. Each model sees only the single prompt. Stateless by design.
  - Runs in batches of BATCH_SIZE. After each batch, prints progress and sleeps
    BATCH_PAUSE_SEC. This prevents continuous hammering of rate limits and makes
    it safe to interrupt between batches.
  - Safe to interrupt at ANY point — completed runs are written to SQLite
    immediately. Re-running fetches only status='pending' rows.
  - Errors are recorded per-run (status='error') — use Cell 4b to retry errors.

Controls:
  MODEL_FILTER  = None      → run all 3 models
                = "gpt4o"   → GPT-5.4 only
                = "claude"  → Claude only
                = "gemini"  → Gemini only
  BATCH_SIZE    = 30        → commit progress report every 30 calls
  BATCH_PAUSE_SEC = 10      → pause between batches (seconds)
"""

MODEL_FILTER    = None   # ← set to model name to run one at a time, None = all pending
BATCH_SIZE      = 30     # calls per batch before progress report + pause
BATCH_PAUSE_SEC = 10     # seconds to pause between batches

con = sqlite3.connect(DB_PATH)
runs_to_do = fetch_pending(con, model_filter=MODEL_FILTER)
con.close()

total = len(runs_to_do)
print(f"Pending runs: {total}  (MODEL_FILTER={MODEL_FILTER!r})")
print(f"Batches: {total // BATCH_SIZE + (1 if total % BATCH_SIZE else 0)}  "
      f"× {BATCH_SIZE} calls, {BATCH_PAUSE_SEC}s pause between batches")
print(f"Each call: independent stateless API request (no session state)")
print("="*65)

done_count  = 0
error_count = 0

for batch_start in range(0, total, BATCH_SIZE):
    batch = runs_to_do[batch_start : batch_start + BATCH_SIZE]
    batch_num = batch_start // BATCH_SIZE + 1
    total_batches = total // BATCH_SIZE + (1 if total % BATCH_SIZE else 0)
    print(f"\n── Batch {batch_num}/{total_batches} "
          f"(runs {batch_start+1}–{batch_start+len(batch)}/{total}) ──")

    for run in batch:
        run_id  = run["run_id"]
        model   = run["model"]
        prompt  = run["full_prompt"]
        call_fn = MODEL_FNS[model]

        global_idx = batch_start + batch.index(run) + 1
        print(f"  [{global_idx}/{total}] {model.upper()} {run_id[:8]}...", end=" ", flush=True)
        try:
            # ── INDEPENDENT API CALL — no history, no session ──
            response_text, model_version = call_fn(prompt)
            now = datetime.now(timezone.utc).isoformat()

            con = sqlite3.connect(DB_PATH)
            con.execute(
                "UPDATE runs SET status='done', model_version=?, completed_at=? WHERE run_id=?",
                (model_version, now, run_id)
            )
            con.execute(
                "INSERT OR IGNORE INTO responses(run_id, raw_response, token_count, responded_at) VALUES(?,?,?,?)",
                (run_id, response_text, len(response_text), now)
            )
            con.commit()
            con.close()

            done_count += 1
            print(f"✓ {len(response_text)}c")

        except Exception as e:
            error_msg = str(e)
            now = datetime.now(timezone.utc).isoformat()
            con = sqlite3.connect(DB_PATH)
            con.execute(
                "UPDATE runs SET status='error', error_message=?, completed_at=? WHERE run_id=?",
                (error_msg[:500], now, run_id)
            )
            con.commit()
            con.close()
            error_count += 1
            print(f"✗ {error_msg[:80]}")

        time.sleep(SLEEP_SEC)

    # ── End of batch: progress summary + pause ────────────────────────────────
    con = sqlite3.connect(DB_PATH)
    total_done = con.execute("SELECT count(*) FROM runs WHERE status='done'").fetchone()[0]
    total_err  = con.execute("SELECT count(*) FROM runs WHERE status='error'").fetchone()[0]
    con.close()
    pct = total_done / 570 * 100
    print(f"\n  ▶ Progress: {total_done}/570 ({pct:.1f}%) done | {total_err} errors")
    if batch_start + BATCH_SIZE < total:
        print(f"  Pausing {BATCH_PAUSE_SEC}s before next batch...")
        time.sleep(BATCH_PAUSE_SEC)

print("\n" + "="*65)
print(f"Session complete. Done this run: {done_count} | Errors: {error_count}")


# %% ── Cell 4b: Retry errors ──────────────────────────────────────────────────
"""
Run this cell separately AFTER Cell 4 if any runs show status='error'.
It resets them to 'pending' and they will be picked up on next run.
"""
con = sqlite3.connect(DB_PATH)
errors = con.execute("SELECT run_id, model, error_message FROM runs WHERE status='error'").fetchall()
if errors:
    print(f"Found {len(errors)} error runs:")
    for run_id, model, err in errors:
        print(f"  {run_id[:8]} | {model} | {err[:80]}")
    reset = input("\nReset all errors to pending? (y/n): ")
    if reset.strip().lower() == "y":
        con.execute("UPDATE runs SET status='pending', error_message=NULL WHERE status='error'")
        con.commit()
        print("Reset done. Re-run Cell 4.")
else:
    print("No error runs. All clean.")
con.close()


# %% ── Cell 5: Status summary ─────────────────────────────────────────────────

con = sqlite3.connect(DB_PATH)
rows = con.execute("""
    SELECT model, status, count(*) as n
    FROM runs
    GROUP BY model, status
    ORDER BY model, status
""").fetchall()
con.close()

print("\nRun status summary:")
print(f"{'Model':<10} {'Status':<10} {'Count':>6}")
print("-"*28)
for row in rows:
    print(f"{row[0]:<10} {row[1]:<10} {row[2]:>6}")

con = sqlite3.connect(DB_PATH)
total_done = con.execute("SELECT count(*) FROM runs WHERE status='done'").fetchone()[0]
con.close()
print(f"\nTotal done: {total_done}/570")
