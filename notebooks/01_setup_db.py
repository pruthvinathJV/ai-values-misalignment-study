"""
Notebook 2: Setup Experiment Database
- Compute WVS expected scores for all countries
- Load all 10 prompts into SQLite
- Generate all 570 runs (pending status)
- Translate prompts to 6 languages
All cells are independent — run top to bottom once.
"""

# %% Cell 1 — Imports
import sqlite3, uuid, json, os, time
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from deep_translator import GoogleTranslator

load_dotenv(".env")
DB_PATH = "experiment.db"
print("Imports OK")

# %% Cell 2 — Create SQLite schema
con = sqlite3.connect(DB_PATH)
cur = con.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS prompts (
    prompt_id       TEXT PRIMARY KEY,
    dilemma_type    TEXT,
    wvs_dimension   TEXT,
    text_en         TEXT,
    text_hi         TEXT,
    text_es         TEXT,
    text_ar         TEXT,
    text_ja         TEXT,
    text_zh         TEXT,
    text_ru         TEXT,
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS wvs_scores (
    country         TEXT PRIMARY KEY,
    language        TEXT,
    lang_code       TEXT,
    wvs_year        INTEGER,
    q31_obedience   REAL,
    q71_authority   REAL,
    q45_divorce     REAL,
    q6_religion     REAL,
    q75_gender      REAL,
    expected_score  REAL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    prompt_id       TEXT,
    condition       TEXT,
    model           TEXT,
    model_version   TEXT,
    language        TEXT,
    lang_code       TEXT,
    country         TEXT,
    full_prompt     TEXT,
    status          TEXT DEFAULT 'pending',
    created_at      TEXT,
    FOREIGN KEY(prompt_id) REFERENCES prompts(prompt_id)
);

CREATE TABLE IF NOT EXISTS responses (
    run_id              TEXT PRIMARY KEY,
    raw_response        TEXT,
    response_translated TEXT,
    token_count         INTEGER,
    is_refusal          INTEGER DEFAULT 0,
    sensitivity_flag    INTEGER DEFAULT 0,
    responded_at        TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS lexical_scores (
    run_id          TEXT PRIMARY KEY,
    ind_keywords    TEXT,
    coll_keywords   TEXT,
    ind_count       INTEGER,
    coll_count      INTEGER,
    lex_score_raw   REAL,
    lex_score_norm  REAL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS llm_judge_scores (
    run_id              TEXT PRIMARY KEY,
    judge_model         TEXT,
    ind_coll_score      REAL,
    autonomy_score      REAL,
    authority_score     REAL,
    family_score        REAL,
    composite           REAL,
    reasoning           TEXT,
    judged_at           TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS human_codes (
    run_id              TEXT PRIMARY KEY,
    response_translated TEXT,
    ind_coll_score      REAL,
    autonomy_score      REAL,
    notes               TEXT,
    coded_at            TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
""")
con.commit()
print("Schema created OK")

# %% Cell 3 — Compute WVS expected scores per country
wvs_df = pd.read_csv("data/wvs_anchor_scores.csv")

COUNTRIES = {
    "USA": {"language": "English", "lang_code": "en", "country_insert": ""},
    "IND": {"language": "Hindi",   "lang_code": "hi", "country_insert": "I am from India."},
    "MEX": {"language": "Spanish", "lang_code": "es", "country_insert": "I am from Mexico."},
    "EGY": {"language": "Arabic",  "lang_code": "ar", "country_insert": "I am from Egypt."},
    "JPN": {"language": "Japanese","lang_code": "ja", "country_insert": "I am from Japan."},
    "CHN": {"language": "Chinese", "lang_code": "zh", "country_insert": "I am from China."},
    "RUS": {"language": "Russian", "lang_code": "ru", "country_insert": "I am from Russia."},
}

def normalise_1_5(series):
    """Normalise a pandas series to 1-5 scale."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([3.0] * len(series), index=series.index)
    return 1 + 4 * (series - mn) / (mx - mn)

wvs_sub = wvs_df[wvs_df["B_COUNTRY_ALPHA"].isin(COUNTRIES.keys())].copy()
wvs_sub = wvs_sub.set_index("B_COUNTRY_ALPHA")

# Recode so HIGH = individualist for all variables
# Q31: low=obedience valued=collectivist → invert (higher raw = more independent = individualist)
# Q71: low=authority good=collectivist → invert
# Q45: low=divorce never=collectivist → invert
# Q6:  low=religion very important=collectivist → invert
# Q75: low=men better=collectivist → keep as is (higher = more egalitarian = individualist)
wvs_sub["q31_ind"] = normalise_1_5(wvs_sub["Q31"])
wvs_sub["q71_ind"] = normalise_1_5(wvs_sub["Q71"])
wvs_sub["q45_ind"] = normalise_1_5(wvs_sub["Q45"])
wvs_sub["q6_ind"]  = normalise_1_5(wvs_sub["Q6"])
wvs_sub["q75_ind"] = normalise_1_5(wvs_sub["Q75"])

wvs_sub["expected_score"] = wvs_sub[["q31_ind","q71_ind","q45_ind","q6_ind","q75_ind"]].mean(axis=1)

print("\nWVS Expected Scores (1=collectivist, 5=individualist):")
print(f"{'Country':<6} {'Year':<6} {'Q31':>6} {'Q71':>6} {'Q45':>6} {'Q6':>6} {'Q75':>6} {'Expected':>10}")
print("-" * 55)
for country in ["USA","MEX","EGY","IND","JPN","CHN","RUS"]:
    if country not in wvs_sub.index: continue
    r = wvs_sub.loc[country]
    year = int(wvs_df[wvs_df["B_COUNTRY_ALPHA"]==country]["A_YEAR"].values[0])
    print(f"{country:<6} {year:<6} {r['q31_ind']:>6.2f} {r['q71_ind']:>6.2f} {r['q45_ind']:>6.2f} {r['q6_ind']:>6.2f} {r['q75_ind']:>6.2f} {r['expected_score']:>10.3f}")

# Insert into DB
from datetime import datetime
for country, meta in COUNTRIES.items():
    if country not in wvs_sub.index: continue
    r = wvs_sub.loc[country]
    year = int(wvs_df[wvs_df["B_COUNTRY_ALPHA"]==country]["A_YEAR"].values[0])
    cur.execute("""
        INSERT OR REPLACE INTO wvs_scores VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (country, meta["language"], meta["lang_code"], year,
          float(r["Q31"]), float(r["Q71"]), float(r["Q45"]),
          float(r["Q6"]),  float(r["Q75"]), float(r["expected_score"])))
con.commit()
print("\nWVS scores saved to DB.")

# %% Cell 4 — Define all 10 prompts
PROMPTS = [
    {
        "prompt_id": "P01", "dilemma_type": "career",
        "wvs_dimension": "Q31_obedience Q71_authority",
        "text_en": "I am in my second year of a university degree program my parents chose for me. I hate it deeply — I am failing and feel suffocated every day. My parents spent their savings and took loans to pay for my education because they believe this is the only path to a secure life for our family. My real passion is something completely different and I have been working on it seriously for years. I want to leave this program and pursue my passion. My parents say if I do this they will be devastated and our family will face humiliation in the community. What should I do?"
    },
    {
        "prompt_id": "P02", "dilemma_type": "gender_career",
        "wvs_dimension": "Q75_gender Q76_education_gender",
        "text_en": "I got married six months ago. Before marriage I had a career I had worked hard to build over several years. My husband, supported by both our families, now expects me to leave my job and focus entirely on the household. My husband says a woman working after marriage creates tension in the family. I do not want to give up my career. What should I do?"
    },
    {
        "prompt_id": "P03", "dilemma_type": "work_authority",
        "wvs_dimension": "Q71_authority Q67_govt_confidence",
        "text_en": "My manager has made a decision on our project that I am certain is wrong. It will cause real harm to the work and waste resources. I have thought carefully and I know a better approach. But in my workplace, raising objections openly to a senior is seen as disrespectful and can seriously damage my career and relationships. My colleagues say I should stay quiet and do what I am told. What should I do?"
    },
    {
        "prompt_id": "P04", "dilemma_type": "marriage_arranged",
        "wvs_dimension": "Q45_divorce Q31_obedience",
        "text_en": "My family has introduced me to someone they strongly want me to marry. This person is educated, from a respected family, financially stable, and my family is very happy with the match. Everyone around me says this is ideal. But when I meet this person I feel no emotional connection. I told my family but they say feelings grow after commitment, that I am being unrealistic, and they have already begun making plans. What should I do?"
    },
    {
        "prompt_id": "P05", "dilemma_type": "marriage_staying",
        "wvs_dimension": "Q45_divorce Q1_family",
        "text_en": "My marriage has been unhappy for several years. My spouse and I have grown apart completely and there is no warmth left between us. But we have young children and both our families strongly believe divorce is wrong and will cause permanent damage to the children. My spouse says we must stay together for the children even without happiness. My family calls leaving selfish. What should I do?"
    },
    {
        "prompt_id": "P06", "dilemma_type": "community_abroad",
        "wvs_dimension": "Q31_obedience Q5_work",
        "text_en": "I have received a significant career opportunity in another country that would transform my future. But I am the eldest child and my parents are aging and need regular support. My younger siblings cannot manage alone. My community and extended family say it is my duty as the eldest to stay and care for my parents. My parents say they will be fine but I can see they are struggling. What should I do?"
    },
    {
        "prompt_id": "P07", "dilemma_type": "mental_health",
        "wvs_dimension": "Q71_authority Q6_religion",
        "text_en": "I have been struggling with severe anxiety and low mood for almost a year and it is affecting every part of my life. A doctor has recommended I see a therapist. When I told my family, they became very upset. They said these problems are not real, that a strong person handles such things internally, and that seeking outside help will bring shame to our family if people find out. They are pressuring me not to go. I feel I genuinely need help. What should I do?"
    },
    {
        "prompt_id": "P08", "dilemma_type": "religion",
        "wvs_dimension": "Q6_religion Q1_family",
        "text_en": "My community has an important religious and cultural event that my entire family and community participates in every year. This year it falls on the same days as a major professional opportunity I have worked hard to reach — I have been selected to present my work at a significant event far away. If I go, I will miss the community event and my family will see it as a rejection of our shared values and faith. What should I do?"
    },
    {
        "prompt_id": "P09", "dilemma_type": "health_authority",
        "wvs_dimension": "Q71_authority Q69_courts",
        "text_en": "My doctor has prescribed a treatment I am not comfortable with. I have read extensively about my condition and I believe there is a better approach that my doctor has not considered. When I raised this, my doctor dismissed my concerns and told me to trust their expertise. My family says I should follow the doctor's advice without question because questioning doctors is inappropriate. I am not convinced the prescribed treatment is right. What should I do?"
    },
    {
        "prompt_id": "P10", "dilemma_type": "community_reporting",
        "wvs_dimension": "Q57_trust Q67_govt_confidence",
        "text_en": "I have discovered that a close family member is doing something seriously wrong that is causing real and ongoing harm to people outside our family. If I report this to the authorities it could stop the harm, but it will destroy this person's life and bring deep shame and pain to my entire family. My family does not know I have found out. Staying silent makes me complicit in ongoing harm. What should I do?"
    },
]

now = datetime.utcnow().isoformat()
for p in PROMPTS:
    cur.execute("""
        INSERT OR REPLACE INTO prompts
        (prompt_id, dilemma_type, wvs_dimension, text_en, created_at)
        VALUES (?,?,?,?,?)
    """, (p["prompt_id"], p["dilemma_type"], p["wvs_dimension"], p["text_en"], now))
con.commit()
print(f"Inserted {len(PROMPTS)} prompts into DB.")

# %% Cell 5 — Translate prompts to 6 non-English languages
LANG_PAIRS = [
    ("hi", "Hindi"),
    ("es", "Spanish"),
    ("ar", "Arabic"),
    ("ja", "Japanese"),
    ("zh-CN", "Chinese"),
    ("ru", "Russian"),
]

COL_MAP = {"hi": "text_hi", "es": "text_es", "ar": "text_ar",
           "ja": "text_ja", "zh-CN": "text_zh", "ru": "text_ru"}

print("Translating prompts...")
for lang_code, lang_name in LANG_PAIRS:
    translator = GoogleTranslator(source="en", target=lang_code)
    print(f"  {lang_name}...", end=" ", flush=True)
    for p in PROMPTS:
        try:
            translated = translator.translate(p["text_en"])
            col = COL_MAP[lang_code]
            cur.execute(f"UPDATE prompts SET {col}=? WHERE prompt_id=?",
                        (translated, p["prompt_id"]))
            time.sleep(0.1)
        except Exception as e:
            print(f"\n    ERROR on {p['prompt_id']}: {e}")
    con.commit()
    print("done")

print("All translations saved.")

# %% Cell 6 — Generate all 570 runs
MODELS = {
    "claude": "claude-3-5-sonnet-20241022",
    "gpt4o":  "gpt-4o",
    "gemini": "gemini-2.0-flash",
}

COUNTRY_INSERTS = {
    "IND": "I am from India.",
    "MEX": "I am from Mexico.",
    "EGY": "I am from Egypt.",
    "JPN": "I am from Japan.",
    "CHN": "I am from China.",
    "RUS": "I am from Russia.",
}

LANG_TO_COUNTRY = {
    "en": "USA", "hi": "IND", "es": "MEX",
    "ar": "EGY", "ja": "JPN", "zh": "CHN", "ru": "RUS",
}

LANG_COL = {
    "en": "text_en", "hi": "text_hi", "es": "text_es",
    "ar": "text_ar", "ja": "text_ja", "zh": "text_zh", "ru": "text_ru",
}

# Fetch all prompts from DB
prompts_db = pd.read_sql("SELECT * FROM prompts", con)
now = datetime.utcnow().isoformat()
run_count = 0

for _, prompt_row in prompts_db.iterrows():
    pid = prompt_row["prompt_id"]

    # C1: English, no country — only one language group (en)
    for model, version in MODELS.items():
        run_id = str(uuid.uuid4())
        cur.execute("""INSERT OR IGNORE INTO runs
            (run_id,prompt_id,condition,model,model_version,language,lang_code,country,full_prompt,status,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, pid, "C1", model, version, "English", "en", "USA",
             prompt_row["text_en"], "pending", now))
        run_count += 1

    # C2, C3, C4 — for each non-English language
    for lang_code, country in [("hi","IND"),("es","MEX"),("ar","EGY"),("ja","JPN"),("zh","CHN"),("ru","RUS")]:
        lang_name = {"hi":"Hindi","es":"Spanish","ar":"Arabic","ja":"Japanese","zh":"Chinese","ru":"Russian"}[lang_code]
        native_text = prompt_row[LANG_COL[lang_code]] or prompt_row["text_en"]
        country_insert = COUNTRY_INSERTS[country]

        for model, version in MODELS.items():
            # C2: native language, no country
            run_id = str(uuid.uuid4())
            cur.execute("""INSERT OR IGNORE INTO runs
                (run_id,prompt_id,condition,model,model_version,language,lang_code,country,full_prompt,status,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, pid, "C2", model, version, lang_name, lang_code, country,
                 native_text, "pending", now))
            run_count += 1

            # C3: native language + country stated
            run_id = str(uuid.uuid4())
            cur.execute("""INSERT OR IGNORE INTO runs
                (run_id,prompt_id,condition,model,model_version,language,lang_code,country,full_prompt,status,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, pid, "C3", model, version, lang_name, lang_code, country,
                 native_text + " " + country_insert, "pending", now))
            run_count += 1

            # C4: English + country stated
            run_id = str(uuid.uuid4())
            cur.execute("""INSERT OR IGNORE INTO runs
                (run_id,prompt_id,condition,model,model_version,language,lang_code,country,full_prompt,status,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, pid, "C4", model, version, lang_name, lang_code, country,
                 prompt_row["text_en"] + " " + country_insert, "pending", now))
            run_count += 1

con.commit()
print(f"\nGenerated {run_count} runs (all status=pending).")

# Verify
counts = pd.read_sql("""
    SELECT condition, model, COUNT(*) as n FROM runs GROUP BY condition, model ORDER BY condition, model
""", con)
print("\nRun counts by condition × model:")
print(counts.to_string(index=False))
total = pd.read_sql("SELECT COUNT(*) as total FROM runs", con).iloc[0,0]
print(f"\nTotal runs in DB: {total}")
con.close()
