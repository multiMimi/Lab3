
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, precision_score,
                              recall_score, f1_score, accuracy_score)

SEED = 8213026
DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# STAGE 1: PRESERVE  (hashes already verified via sha256sum -c on originals)
# ---------------------------------------------------------------------------
assets = pd.read_csv(DATA / "egs_asset_inventory.csv")
edges = pd.read_csv(DATA / "egs_network_topology_edges.csv")
posture = pd.read_csv(DATA / "egs_host_security_posture.csv")
scenarios = pd.read_csv(DATA / "egs_simulation_scenarios.csv")
events = pd.read_csv(DATA / "egs_security_event_logs.csv", parse_dates=["timestamp"])
ioc_feed = pd.read_csv(DATA / "egs_ioc_feed.csv", parse_dates=["first_seen"])
text_train = pd.read_csv(DATA / "egs_text_training.csv", parse_dates=["published_at"])
text_investigation = pd.read_csv(DATA / "egs_text_investigation.csv", parse_dates=["published_at"])
adv_text = pd.read_csv(DATA / "egs_adversarial_text_cases.csv")
risk_train = pd.read_csv(DATA / "egs_daily_risk_training.csv", parse_dates=["date"])
risk_investigation = pd.read_csv(DATA / "egs_daily_risk_investigation.csv", parse_dates=["date"])
adv_risk = pd.read_csv(DATA / "egs_adversarial_risk_cases.csv")

print("Row counts:", {"assets": len(assets), "edges": len(edges), "events": len(events),
                       "text_train": len(text_train), "text_investigation": len(text_investigation),
                       "risk_train": len(risk_train), "risk_investigation": len(risk_investigation)})

# ---------------------------------------------------------------------------
# PART B1: TEXT PREPARATION & EXPLORATORY ANALYSIS
# ---------------------------------------------------------------------------
print("\n--- B1: Text EDA ---")
for name, df in [("text_train", text_train), ("text_investigation", text_investigation)]:
    print(f"\n{name}: missing values\n", df.isna().sum())
    print(f"{name}: duplicate report_id count:", df["report_id"].duplicated().sum())
    print(f"{name}: duplicate report_text count:", df["report_text"].duplicated().sum())

print("\nSource distribution (train):\n", text_train["source"].value_counts())
print("\nClass balance (train relevant_label):\n", text_train["relevant_label"].value_counts(normalize=True))
print("\nDate range (train):", text_train["published_at"].min(), "to", text_train["published_at"].max())
print("Date range (investigation):", text_investigation["published_at"].min(), "to", text_investigation["published_at"].max())

# --- Text cleaning pipeline ---
def clean_text(s: str) -> str:
    """Lowercase, strip non-alphanumeric (keep IOC-relevant chars for a separate
    raw-IOC pass later), collapse whitespace. Used for the TF-IDF classifier only;
    IOC extraction runs on the RAW text separately so punctuation isn't lost."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\.\-:/]", " ", s)   # keep . - : / for now (helps entity-ish tokens); TF-IDF token pattern below finishes the job
    s = re.sub(r"\s+", " ", s).strip()
    return s

text_train["clean_text"] = (text_train["title"] + " " + text_train["report_text"]).apply(clean_text)
text_investigation["clean_text"] = (text_investigation["title"] + " " + text_investigation["report_text"]).apply(clean_text)

# --- B1 visualisations ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
text_train["source"].value_counts().plot(kind="bar", ax=axes[0], color="#2b6cb0")
axes[0].set_title("Training reports by source")
axes[0].set_ylabel("count")

text_train.set_index("published_at").resample("ME")["relevant_label"].mean().plot(ax=axes[1], color="#c05621")
axes[1].set_title("Monthly mean relevance rate (training)")
axes[1].set_ylabel("proportion relevant")
plt.tight_layout()
plt.savefig(OUT / "b1_source_time_distribution.png", dpi=120)
plt.close()

# term frequency by class (top unigrams, relevant vs not) - simple, transparent
from collections import Counter
STOP = set("""a an the of to in and or for is are was were be been being with on at by from as it its this that
these those will would could should can may might must not no nor but if then than so such into over under
between about after before during while has have had do does did report reports""".split())

def top_terms(texts, n=15):
    c = Counter()
    for t in texts:
        for w in t.split():
            if len(w) > 2 and w not in STOP and not w.isdigit():
                c[w] += 1
    return c.most_common(n)

relevant_terms = top_terms(text_train[text_train.relevant_label==1]["clean_text"])
nonrelevant_terms = top_terms(text_train[text_train.relevant_label==0]["clean_text"])
term_table = pd.DataFrame({
    "relevant_top_terms": [f"{w} ({c})" for w,c in relevant_terms],
    "nonrelevant_top_terms": [f"{w} ({c})" for w,c in nonrelevant_terms],
})
term_table.to_csv(OUT / "b1_top_terms_by_class.csv", index=False)
print("\n--- B1 top terms by class ---")
print(term_table)

# ---------------------------------------------------------------------------
# PART B2: IOC / ENTITY EXTRACTION
# ---------------------------------------------------------------------------
print("\n--- B2: Entity extraction ---")

IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
DOMAIN_RE = re.compile(r'\b[a-z0-9][a-z0-9\-]*\.[a-z]{2,}\b')
HASH_RE = re.compile(r'\b[a-f0-9]{20,64}\b')
ACTOR_RE = re.compile(r'\bactor ([A-Z][\w]*(?:\s[A-Z][\w]*){0,2})\b')
MALWARE_RE = re.compile(r'\b([A-Z][a-z]+ (?:Loader|Stealer|Wiper|Backdoor|RAT|Miner|Trojan))\b')

def extract_entities(report_id, raw_text):
    rows = []
    for ip in set(IP_RE.findall(raw_text)):
        rows.append((report_id, "ipv4", ip))
    for d in set(DOMAIN_RE.findall(raw_text)):
        if not IP_RE.fullmatch(d):
            rows.append((report_id, "domain", d))
    for h in set(HASH_RE.findall(raw_text)):
        rows.append((report_id, "hash", h))
    for a in set(ACTOR_RE.findall(raw_text)):
        if a.lower() != "unknown":
            rows.append((report_id, "threat_actor", a))
    for m in set(MALWARE_RE.findall(raw_text)):
        rows.append((report_id, "malware", m))
    return rows

all_entity_rows = []
combined_text_df = pd.concat([
    text_train[["report_id", "title", "report_text"]],
    text_investigation[["report_id", "title", "report_text"]]
], ignore_index=True)

for _, row in combined_text_df.iterrows():
    all_entity_rows.extend(extract_entities(row["report_id"], f'{row["title"]} {row["report_text"]}'))

extracted_iocs = pd.DataFrame(all_entity_rows, columns=["report_id", "entity_type", "entity_value"])

# Match against ioc_feed for operational relevance (confidence/recency/context)
ioc_feed_lookup = ioc_feed.copy()
ioc_feed_lookup["indicator_value_lower"] = ioc_feed_lookup["indicator_value"].str.lower()
extracted_iocs["entity_value_lower"] = extracted_iocs["entity_value"].str.lower()

def match_feed(val, etype):
    # ipv4/hash/domain: exact match only. URLs in the feed may embed a domain,
    # so allow substring match specifically for entity_type == "domain" against feed "url" rows.
    exact = ioc_feed_lookup[ioc_feed_lookup["indicator_value_lower"] == val]
    if len(exact):
        h = exact.iloc[0]
        return pd.Series([h["ioc_id"], h["confidence"], h["first_seen"], h["tactic"]])
    if etype == "domain":
        contained = ioc_feed_lookup[ioc_feed_lookup["indicator_value_lower"].str.contains(re.escape(val), regex=True)]
        if len(contained):
            h = contained.iloc[0]
            return pd.Series([h["ioc_id"], h["confidence"], h["first_seen"], h["tactic"]])
    return pd.Series([None, None, None, None])

extracted_iocs[["matched_ioc_id", "feed_confidence", "feed_first_seen", "feed_tactic"]] = \
    extracted_iocs.apply(lambda r: match_feed(r["entity_value_lower"], r["entity_type"]), axis=1)

extracted_iocs["analytical_relevance"] = np.select(
    [extracted_iocs["matched_ioc_id"].notna() & (extracted_iocs["feed_confidence"] == "High"),
     extracted_iocs["matched_ioc_id"].notna() & (extracted_iocs["feed_confidence"] == "Medium"),
     extracted_iocs["matched_ioc_id"].notna()],
    ["High - matched external IOC (high confidence)",
     "Medium - matched external IOC (medium confidence)",
     "Low - matched external IOC (low confidence)"],
    default="Unconfirmed - internal-only entity, no external IOC match"
)

extracted_iocs = extracted_iocs.drop(columns=["entity_value_lower"])
extracted_iocs.to_csv(OUT / "extracted_iocs.csv", index=False)
print("extracted_iocs.csv rows:", len(extracted_iocs))
print(extracted_iocs["entity_type"].value_counts())
print("\nMatched to external feed:", extracted_iocs["matched_ioc_id"].notna().sum(), "of", len(extracted_iocs))
print(extracted_iocs[extracted_iocs["matched_ioc_id"].notna()].head(8).to_string())

# ---------------------------------------------------------------------------
# PART B3: TEXT CLASSIFIER & ADVERSARIAL LANGUAGE
# ---------------------------------------------------------------------------
print("\n--- B3: Text classifier ---")

X_train_full, y_train_full = text_train["clean_text"], text_train["relevant_label"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=SEED
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_tr_vec = vectorizer.fit_transform(X_tr)
X_te_vec = vectorizer.transform(X_te)

clf = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
clf.fit(X_tr_vec, y_tr)
y_pred = clf.predict(X_te_vec)

cm = confusion_matrix(y_te, y_pred)
tn, fp, fn, tp = cm.ravel()
fnr = fn / (fn + tp)
print("Confusion matrix [ [TN FP] [FN TP] ]:\n", cm)
print(f"Accuracy: {accuracy_score(y_te, y_pred):.3f}")
print(f"Precision: {precision_score(y_te, y_pred):.3f}")
print(f"Recall: {recall_score(y_te, y_pred):.3f}")
print(f"F1: {f1_score(y_te, y_pred):.3f}")
print(f"False-negative rate: {fnr:.3f}")
print(classification_report(y_te, y_pred))

# Score investigation reports
X_inv_vec = vectorizer.transform(text_investigation["clean_text"])
text_investigation["relevance_score"] = clf.predict_proba(X_inv_vec)[:, 1]
top_15 = text_investigation.sort_values("relevance_score", ascending=False).head(15)
top_15[["report_id", "published_at", "source", "title", "relevance_score"]].to_csv(
    OUT / "top_15_relevant_reports.csv", index=False
)
print("\nTop 15 investigation reports by relevance score:")
print(top_15[["report_id", "source", "relevance_score"]].to_string())

# Adversarial text comparison
print("\n--- Adversarial text cases ---")
adv_results = []
for _, row in adv_text.iterrows():
    orig_clean = clean_text(row["original_text"])
    mod_clean = clean_text(row["modified_text"])
    orig_score = clf.predict_proba(vectorizer.transform([orig_clean]))[0, 1]
    mod_score = clf.predict_proba(vectorizer.transform([mod_clean]))[0, 1]
    adv_results.append({
        "case_id": row["case_id"], "evasion_technique": row["evasion_technique"],
        "original_score": orig_score, "modified_score": mod_score,
        "score_change": mod_score - orig_score
    })
adv_df = pd.DataFrame(adv_results)
print(adv_df.to_string())

# ---------------------------------------------------------------------------
# PART C1: SIMULATION MODEL DESIGN
# ---------------------------------------------------------------------------
print("\n--- C1/C2: Propagation simulation ---")

G = nx.from_pandas_edgelist(edges, "source_asset", "target_asset", edge_attr=True, create_using=nx.DiGraph)
susceptibility_map = dict(zip(posture["asset_id"], posture["susceptibility"]))
criticality_map = dict(zip(assets["asset_id"], assets["criticality"]))

SEED_NODE = "VENDOR-LT-07"
VENDOR_PATH_ASSETS = {"VENDOR-LT-07", "VPN-GW-01", "VENDOR-DMZ-01"}  # scope of "patch vendor path" scenario
MAX_STEPS = 40
N_ITER = 1000

def effective_prob(u, v, edata, scenario, step):
    p = edata["base_transmission_probability"]
    if u in VENDOR_PATH_ASSETS:
        p *= scenario["patch_factor"]
    # segmentation control governs DMZ->OT entry and lateral movement within OT
    if edata["target_zone"] in ("OT-DMZ", "OT") and (edata["source_zone"] != edata["target_zone"] or edata["target_zone"] == "OT"):
        p *= scenario["segmentation_factor"]
    p *= susceptibility_map.get(v, 1.0)
    if step > scenario["isolation_delay_steps"]:
        p *= scenario["post_detection_transmission_factor"]
    return min(p, 1.0)

def run_single_sim(scenario, rng):
    infected = {SEED_NODE}
    infected_time = {SEED_NODE: 0}
    active = [SEED_NODE]  # deterministic order: process nodes in a fixed sequence, not set iteration order
    for step in range(1, MAX_STEPS + 1):
        new_infections = []
        for u in active:
            for v in sorted(G.successors(u)):  # sorted for reproducibility across runs/platforms
                if v in infected:
                    continue
                p = effective_prob(u, v, G.edges[u, v], scenario, step)
                if rng.random() < p and v not in infected:
                    new_infections.append(v)
                    infected.add(v)
        if not new_infections:
            break
        for v in new_infections:
            infected_time[v] = step
        active = new_infections
    return infected, infected_time

zone_map = dict(zip(assets["asset_id"], assets["zone"]))
OT_NODES = {a for a, z in zone_map.items() if z == "OT"}
CRITICAL_NODES = {a for a, c in criticality_map.items() if c == "Critical"}
SAFETY_NODE = "SAFETY-PLC-01"

def summarize_scenario(scenario_row, n_iter=N_ITER, seed=SEED):
    rng = np.random.default_rng(seed)
    reached_ot, reached_safety = [], []
    critical_infected_counts, total_infected_counts, time_to_ot = [], [], []
    for _ in range(n_iter):
        infected, infected_time = run_single_sim(scenario_row, rng)
        ot_hit = bool(infected & OT_NODES)
        reached_ot.append(ot_hit)
        reached_safety.append(SAFETY_NODE in infected)
        critical_infected_counts.append(len(infected & CRITICAL_NODES))
        total_infected_counts.append(len(infected))
        if ot_hit:
            ot_times = [infected_time[n] for n in (infected & OT_NODES)]
            time_to_ot.append(min(ot_times))
    total_infected_counts = np.array(total_infected_counts)
    return {
        "scenario_id": scenario_row["scenario_id"],
        "scenario_name": scenario_row["scenario_name"],
        "n_iterations": n_iter,
        "prob_reach_OT": np.mean(reached_ot),
        "prob_reach_safety_zone": np.mean(reached_safety),
        "mean_critical_assets_infected": np.mean(critical_infected_counts),
        "mean_total_assets_infected": np.mean(total_infected_counts),
        "median_time_to_OT_steps": np.median(time_to_ot) if time_to_ot else np.nan,
        "p95_total_assets_infected": np.percentile(total_infected_counts, 95),
    }

sim_summary = pd.DataFrame([summarize_scenario(row) for _, row in scenarios.iterrows()])

baseline = sim_summary[sim_summary.scenario_id == "S0"].iloc[0]
sim_summary["abs_risk_reduction_OT"] = baseline["prob_reach_OT"] - sim_summary["prob_reach_OT"]
sim_summary["rel_risk_reduction_OT_pct"] = (sim_summary["abs_risk_reduction_OT"] / baseline["prob_reach_OT"] * 100).round(1)

sim_summary.to_csv(OUT / "simulation_summary.csv", index=False)
print(sim_summary.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(sim_summary["scenario_name"], sim_summary["prob_reach_OT"], color="#9b2c2c")
axes[0].set_title("Probability of reaching OT zone, by control scenario")
axes[0].set_ylabel("P(reach OT)")
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(sim_summary["scenario_name"], sim_summary["mean_critical_assets_infected"], color="#2c5282")
axes[1].set_title("Mean critical assets infected, by control scenario")
axes[1].set_ylabel("mean count (of 1000 runs)")
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(OUT / "c2_scenario_comparison.png", dpi=120)
plt.close()
print("\nSaved C2 comparison chart.")

# ---------------------------------------------------------------------------
# PART D1: PREDICTIVE RISK MODEL
# ---------------------------------------------------------------------------
print("\n--- D1: Predictive risk model ---")

FEATURE_COLS = [c for c in risk_train.columns if c not in ("date", "incident_within_7d")]
Xr, yr = risk_train[FEATURE_COLS], risk_train["incident_within_7d"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, stratify=yr, random_state=SEED)

rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=SEED)
rf.fit(Xr_tr, yr_tr)

# Default threshold 0.5 first
yr_proba = rf.predict_proba(Xr_te)[:, 1]
yr_pred_50 = (yr_proba >= 0.5).astype(int)
cm50 = confusion_matrix(yr_te, yr_pred_50)
print("Threshold=0.50 confusion matrix:\n", cm50)
print(f"  Acc={accuracy_score(yr_te,yr_pred_50):.3f} Prec={precision_score(yr_te,yr_pred_50):.3f} "
      f"Rec={recall_score(yr_te,yr_pred_50):.3f} F1={f1_score(yr_te,yr_pred_50):.3f} "
      f"FNR={cm50[1,0]/(cm50[1,0]+cm50[1,1]):.3f}")

# Lower threshold to prioritise recall (fewer missed incidents) for critical infrastructure
THRESH = 0.35
yr_pred_lo = (yr_proba >= THRESH).astype(int)
cm_lo = confusion_matrix(yr_te, yr_pred_lo)
print(f"\nThreshold={THRESH} confusion matrix:\n", cm_lo)
print(f"  Acc={accuracy_score(yr_te,yr_pred_lo):.3f} Prec={precision_score(yr_te,yr_pred_lo):.3f} "
      f"Rec={recall_score(yr_te,yr_pred_lo):.3f} F1={f1_score(yr_te,yr_pred_lo):.3f} "
      f"FNR={cm_lo[1,0]/(cm_lo[1,0]+cm_lo[1,1]):.3f}")

feat_importance = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\nFeature importances:\n", feat_importance)

# ---------------------------------------------------------------------------
# PART D2: RISK ESCALATION AND CORRELATION
# ---------------------------------------------------------------------------
print("\n--- D2: Score investigation days ---")
risk_investigation["risk_probability"] = rf.predict_proba(risk_investigation[FEATURE_COLS])[:, 1]
top_10_days = risk_investigation.sort_values("risk_probability", ascending=False).head(10)
top_10_days[["date", "risk_probability"] + FEATURE_COLS].to_csv(OUT / "top_10_risk_days.csv", index=False)
print(risk_investigation[["date", "risk_probability"]].to_string())

escalation_point = risk_investigation[risk_investigation["risk_probability"] >= THRESH].sort_values("date").head(1)
print("\nEarliest date crossing operating threshold (%.2f):" % THRESH)
print(escalation_point[["date", "risk_probability"]])

plt.figure(figsize=(9, 4))
plt.plot(risk_investigation["date"], risk_investigation["risk_probability"], marker="o", color="#9b2c2c")
plt.axhline(THRESH, color="gray", linestyle="--", label=f"operating threshold ({THRESH})")
plt.title("Daily incident-risk probability, investigation period")
plt.ylabel("P(incident within 7 days)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "d2_risk_probability_timeline.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# PART D3: ADVERSARIAL ROBUSTNESS
# ---------------------------------------------------------------------------
print("\n--- D3: Adversarial risk cases ---")
adv_risk["risk_probability"] = rf.predict_proba(adv_risk[FEATURE_COLS])[:, 1]
adv_risk["flagged_at_threshold"] = adv_risk["risk_probability"] >= THRESH
print(adv_risk[["case_id", "variant", "risk_probability", "flagged_at_threshold"]].to_string())

baseline_prob = adv_risk.loc[adv_risk.variant=="baseline_attack", "risk_probability"].values[0]
adv_risk["prob_drop_vs_baseline"] = baseline_prob - adv_risk["risk_probability"]
print("\nProbability drop vs full-signature baseline attack:")
print(adv_risk[["case_id","variant","prob_drop_vs_baseline"]].to_string())
