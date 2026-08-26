from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
UPLOAD = ROOT / "upload"
OUT = ROOT / "analysis_output"
OUT.mkdir(exist_ok=True)

CLINICAL_FILE = UPLOAD / "ARG DATA (0-3)(20260826-103813).xlsx"
COUNTS_FILE = UPLOAD / "ASV_table_counts(20260826-103801).csv"
REL_FILE = UPLOAD / "ASV_table_relative_abundance(20260826-103802).csv"

ANTIBIOTIC_EXCLUSIONS = {150, 168, 271}
N_PERM = 999

IMMUNE = [
    "IL-1 beta",
    "IL1-RA",
    "IL-10",
    "IL-22",
    "MCP-1 (CCL2)",
    "TNF-alpha",
    "IL-6",
    "IL-8",
]
DIET = [
    "Global_Diet_Quality",
    "Probiotic_Exposure",
    "Sweetened_Beverage_Load",
    "Fat_Quality_Index",
    "Protein_Profile",
    "Supplementation_Score",
]
PA = "TOTAL_MET_minwk"
HOMA = "HOMA-IR"


def bh_adjust(values):
    p = np.asarray(values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    out[ok] = restored
    return out


def safe_z(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return x * np.nan
    return (x - np.nanmean(x)) / sd


def transform_series(s, kind="identity"):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if kind == "log1p":
        return np.log1p(np.clip(x, a_min=0, a_max=None))
    if kind == "log":
        return np.log(np.clip(x, a_min=1e-12, a_max=None))
    return x


def fit_hc3(y, data, effect, specs, effect_binary=False):
    """OLS with HC3 SE. Continuous columns are standardized; binary columns are not."""
    frame = pd.DataFrame({"__y": pd.to_numeric(y, errors="coerce")}, index=data.index)
    for name, arr, binary in specs:
        frame[name] = pd.to_numeric(pd.Series(arr, index=data.index), errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < len(specs) + 6 or frame["__y"].nunique() < 2:
        return None
    yz = safe_z(frame["__y"].to_numpy())
    if not np.isfinite(yz).all():
        return None
    cols = []
    names = []
    for name, _arr, binary in specs:
        x = frame[name].to_numpy(dtype=float)
        if not binary:
            x = safe_z(x)
        if not np.isfinite(x).all() or np.nanstd(x) == 0:
            return None
        cols.append(x)
        names.append(name)
    X = np.column_stack([np.ones(len(frame))] + cols)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = np.linalg.lstsq(X, yz, rcond=None)[0]
    resid = yz - X @ beta
    hat = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    denom = np.clip(1.0 - hat, 1e-8, None)
    u = resid / denom
    meat = X.T @ ((u * u)[:, None] * X)
    cov = xtx_inv @ meat @ xtx_inv
    j = 1 + names.index(effect)
    se = float(np.sqrt(max(cov[j, j], 0)))
    df = len(frame) - X.shape[1]
    if se <= 0 or df <= 0:
        return None
    tval = float(beta[j] / se)
    pval = float(2 * stats.t.sf(abs(tval), df))
    crit = float(stats.t.ppf(0.975, df))
    return {
        "n": int(len(frame)),
        "beta": float(beta[j]),
        "se_hc3": se,
        "ci_low": float(beta[j] - crit * se),
        "ci_high": float(beta[j] + crit * se),
        "p": pval,
        "df": int(df),
    }


def spearman_pair(x, y):
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 5 or z.x.nunique() < 2 or z.y.nunique() < 2:
        return np.nan, np.nan
    r = stats.spearmanr(z.x, z.y)
    return float(r.statistic), float(r.pvalue)


def add_fdr(df, group_cols=("family",), p_col="p", q_col="q"):
    df = df.copy()
    df[q_col] = np.nan
    if len(df) == 0:
        return df
    for _, idx in df.groupby(list(group_cols), dropna=False).groups.items():
        df.loc[idx, q_col] = bh_adjust(df.loc[idx, p_col].to_numpy())
    return df


def clean_clinical():
    d = pd.read_excel(CLINICAL_FILE)
    d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
    d = d.drop(columns=[c for c in d.columns if str(c).strip() == "l"], errors="ignore")
    d.columns = [str(c).strip() for c in d.columns]
    d = d[d["timeline"].isin([0, 1])].copy()
    d = d.rename(columns={"TNF‑α": "TNF-alpha", "TNF‑α ": "TNF-alpha"})
    # Excel contains one decimal-comma IL-6 entry ("7,3").
    d["IL-6"] = pd.to_numeric(d["IL-6"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    numeric = [
        "Code #", "Intervention", "timeline", "Age", "BMI", "family history of diabetes",
        HOMA, PA, *IMMUNE, *DIET,
    ]
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Code #"] = d["Code #"].astype(int)
    d["timeline"] = d["timeline"].astype(int)
    d["Intervention_label"] = d["Intervention"].map({1: "Placebo", 2: "Synbiotic"})
    d["Group"] = (d["Intervention"] == 2).astype(int)
    d["Sex_F"] = d["Gender"].astype(str).str.upper().str.startswith("F").astype(int)
    d["FHD_binary"] = d["family history of diabetes"].where(d["family history of diabetes"].isin([0, 1]))
    d["log_MET"] = np.log1p(np.clip(d[PA], 0, None))
    d["log_HOMA"] = np.log(np.clip(d[HOMA], 1e-12, None))
    for c in IMMUNE:
        d[f"log_{c}"] = np.log1p(np.clip(d[c], 0, None))
    return d


def make_wide(d):
    b = d[d.timeline == 0].set_index("Code #").copy()
    e = d[d.timeline == 1].set_index("Code #").copy()
    codes = sorted(set(b.index) & set(e.index))
    b = b.loc[codes]
    e = e.loc[codes]
    w = pd.DataFrame(index=codes)
    static = ["Participant ID", "Intervention", "Intervention_label", "Group", "Gender", "Sex_F", "Age", "BMI", "FHD_binary", "family history of diabetes"]
    for c in static:
        w[f"{c}_0"] = b[c]
    w["Participant ID_1"] = e["Participant ID"]
    variables = [HOMA, "log_HOMA", PA, "log_MET", *DIET, *IMMUNE] + [f"log_{c}" for c in IMMUNE]
    for c in variables:
        w[f"{c}_0"] = b[c]
        w[f"{c}_1"] = e[c]
        w[f"d_{c}"] = e[c] - b[c]
    return b, e, w


def microbiome_objects(d, b, e, w):
    counts = pd.read_csv(COUNTS_FILE).set_index("Feature")
    rel = pd.read_csv(REL_FILE).set_index("Feature")
    # Keep only valid paired clinical subjects with both microbiome visits.
    paired = []
    for code in w.index:
        s0 = str(w.loc[code, "Participant ID_0"])
        s1 = str(w.loc[code, "Participant ID_1"])
        if s0 in counts.columns and s1 in counts.columns:
            paired.append(code)
    sensitivity_codes = paired
    primary_codes = [c for c in paired if c not in ANTIBIOTIC_EXCLUSIONS]
    primary_baseline_samples = [str(w.loc[c, "Participant ID_0"]) for c in primary_codes]
    prevalence = (counts[primary_baseline_samples] > 0).mean(axis=1)
    taxa = prevalence[prevalence >= 0.20].index.tolist()
    selected_samples = []
    for c in sensitivity_codes:
        selected_samples += [str(w.loc[c, "Participant ID_0"]), str(w.loc[c, "Participant ID_1"])]
    rr = rel.loc[taxa, selected_samples].astype(float).copy()
    # Taxon-wise half-minimum replacement followed by closure and CLR.
    for taxon in taxa:
        row = rr.loc[taxon].to_numpy(dtype=float)
        positive = row[row > 0]
        replacement = 0.5 * positive.min() if len(positive) else 1e-12
        row[row <= 0] = replacement
        rr.loc[taxon] = row
    rr = rr / rr.sum(axis=0)
    log_rr = np.log(rr)
    clr = log_rr.subtract(log_rr.mean(axis=0), axis=1).T

    # Alpha diversity uses all observed features from the count table.
    cc = counts[selected_samples].T.astype(float)
    total = cc.sum(axis=1)
    probs = cc.div(total, axis=0)
    alpha = pd.DataFrame(index=cc.index)
    alpha["Observed_richness"] = (cc > 0).sum(axis=1)
    alpha["Shannon"] = -(probs.where(probs > 0) * np.log(probs.where(probs > 0))).sum(axis=1)
    alpha["Simpson"] = 1 - (probs * probs).sum(axis=1)

    # Derive a genus-level table from the supplied species labels. Bracketed legacy
    # genus names (for example [Eubacterium]) are normalized by removing brackets.
    genus_map = pd.Series({t: str(t).split()[0].strip("[]") for t in rel.index})
    genus_rel = rel.copy()
    genus_rel["__genus"] = genus_map
    genus_rel = genus_rel.groupby("__genus", sort=True).sum(numeric_only=True)
    genus_counts = counts.copy()
    genus_counts["__genus"] = genus_map
    genus_counts = genus_counts.groupby("__genus", sort=True).sum(numeric_only=True)
    genus_prevalence = (genus_counts[primary_baseline_samples] > 0).mean(axis=1)
    genera = genus_prevalence[genus_prevalence >= 0.20].index.tolist()
    gr = genus_rel.loc[genera, selected_samples].astype(float).copy()
    for genus in genera:
        row = gr.loc[genus].to_numpy(dtype=float)
        positive = row[row > 0]
        replacement = 0.5 * positive.min() if len(positive) else 1e-12
        row[row <= 0] = replacement
        gr.loc[genus] = row
    gr = gr / gr.sum(axis=0)
    genus_log = np.log(gr)
    genus_clr = genus_log.subtract(genus_log.mean(axis=0), axis=1).T

    return (counts, rel, clr, alpha, taxa, primary_codes, sensitivity_codes, prevalence,
            genus_clr, genera, genus_prevalence)


def baseline_nonmicro(b):
    rows = []
    age = b["Age"]
    sex = b["Sex_F"]
    bmi = b["BMI"]

    def run(family, predictor, pred_values, outcome, out_values, covs, pred_binary=False, model_note=""):
        specs = [(predictor, pred_values, pred_binary)] + covs
        fit = fit_hc3(out_values, b, predictor, specs, effect_binary=pred_binary)
        if fit is None:
            return
        rho, rho_p = spearman_pair(pred_values, out_values)
        rows.append({"analysis": "Baseline", "family": family, "predictor": predictor, "outcome": outcome,
                     "rho": rho, "rho_p": rho_p, **fit, "model": model_note})

    core_covs = [("Age", age, False), ("Sex_F", sex, True), ("BMI", bmi, False)]
    diet_covs = core_covs + [("log_MET", b["log_MET"], False)]

    for y in IMMUNE:
        run("PA-immune", "Physical activity (log MET-min/wk)", b["log_MET"], y, b[f"log_{y}"], core_covs,
            model_note="log1p(cytokine) ~ log1p(MET) + age + sex + BMI")
    for x in DIET:
        for y in IMMUNE:
            run("Diet-immune", x, b[x], y, b[f"log_{y}"], diet_covs,
                model_note="log1p(cytokine) ~ diet score + age + sex + BMI + log1p(MET)")
    for y in IMMUNE:
        run("FHD-immune", "Family history (yes vs no)", b["FHD_binary"], y, b[f"log_{y}"], core_covs, True,
            "log1p(cytokine) ~ family history + age + sex + BMI; code 2 excluded")

    run("Lifestyle-HOMA", "Physical activity (log MET-min/wk)", b["log_MET"], HOMA, b["log_HOMA"], core_covs,
        model_note="log(HOMA-IR) ~ log1p(MET) + age + sex + BMI")
    for x in DIET:
        run("Lifestyle-HOMA", x, b[x], HOMA, b["log_HOMA"], diet_covs,
            model_note="log(HOMA-IR) ~ diet score + age + sex + BMI + log1p(MET)")
    for x in IMMUNE:
        run("Immune-HOMA", x, b[f"log_{x}"], HOMA, b["log_HOMA"], core_covs + [("log_MET", b["log_MET"], False)],
            model_note="log(HOMA-IR) ~ log1p(cytokine) + age + sex + BMI + log1p(MET)")
    run("FHD-HOMA", "Family history (yes vs no)", b["FHD_binary"], HOMA, b["log_HOMA"], core_covs, True,
        "log(HOMA-IR) ~ family history + age + sex + BMI; code 2 excluded")

    run("FHD-lifestyle", "Family history (yes vs no)", b["FHD_binary"], "Physical activity (log MET-min/wk)", b["log_MET"], core_covs, True,
        "log1p(MET) ~ family history + age + sex + BMI; code 2 excluded")
    for y in DIET:
        run("FHD-lifestyle", "Family history (yes vs no)", b["FHD_binary"], y, b[y], core_covs, True,
            "diet score ~ family history + age + sex + BMI; code 2 excluded")
        run("PA-diet", "Physical activity (log MET-min/wk)", b["log_MET"], y, b[y], core_covs,
            model_note="diet score ~ log1p(MET) + age + sex + BMI")

    out = pd.DataFrame(rows)
    out = add_fdr(out, ("family",), "p", "q_family")
    out = add_fdr(out, ("family",), "rho_p", "q_spearman")
    out["evidence"] = np.select(
        [out.q_family < 0.05, out.q_spearman < 0.05, out.p < 0.05],
        ["FDR-significant adjusted", "FDR-significant Spearman only", "Nominal only"],
        default="Not significant")
    return out.sort_values(["q_family", "p"])


def baseline_microbiome(b, w, clr, alpha, taxa, primary_codes):
    rows_taxa, rows_alpha = [], []
    codes = primary_codes
    clin = b.loc[codes].copy()
    sample_ids = [str(w.loc[c, "Participant ID_0"]) for c in codes]
    Y = clr.loc[sample_ids, taxa].copy()
    Y.index = codes
    A = alpha.loc[sample_ids].copy()
    A.index = codes
    age, sex, bmi = clin.Age, clin.Sex_F, clin.BMI
    core_covs = [("Age", age, False), ("Sex_F", sex, True), ("BMI", bmi, False)]

    exposures = []
    exposures.append(("PA-taxa", "Physical activity (log MET-min/wk)", clin.log_MET, False, core_covs))
    for x in DIET:
        exposures.append(("Diet-taxa", x, clin[x], False, core_covs + [("log_MET", clin.log_MET, False)]))
    for x in IMMUNE:
        exposures.append(("Immune-taxa", x, clin[f"log_{x}"], False, core_covs + [("log_MET", clin.log_MET, False)]))
    exposures.append(("FHD-taxa", "Family history (yes vs no)", clin.FHD_binary, True, core_covs))
    exposures.append(("HOMA-taxa", HOMA, clin.log_HOMA, False, core_covs + [("log_MET", clin.log_MET, False)]))

    for family, pred, x, binary, covs in exposures:
        for taxon in taxa:
            fit = fit_hc3(Y[taxon], clin, pred, [(pred, x, binary)] + covs)
            if fit:
                rho, rho_p = spearman_pair(x, Y[taxon])
                rows_taxa.append({"analysis": "Baseline", "family": family, "predictor": pred, "taxon": taxon,
                                  "rho": rho, "rho_p": rho_p, **fit})
        for metric in A.columns:
            fit = fit_hc3(A[metric], clin, pred, [(pred, x, binary)] + covs)
            if fit:
                rho, rho_p = spearman_pair(x, A[metric])
                rows_alpha.append({"analysis": "Baseline", "family": family.replace("taxa", "alpha"),
                                   "predictor": pred, "outcome": metric, "rho": rho, "rho_p": rho_p, **fit})

    tax = pd.DataFrame(rows_taxa)
    tax = add_fdr(tax, ("family",), "p", "q_domain")
    tax = add_fdr(tax, ("family", "predictor"), "p", "q_predictor")
    tax = add_fdr(tax, ("family",), "rho_p", "q_spearman_domain")
    tax = add_fdr(tax, ("family", "predictor"), "rho_p", "q_spearman_predictor")
    tax["evidence"] = np.select([tax.q_domain < 0.05, tax.q_predictor < 0.05,
                                 tax.q_spearman_domain < 0.05, tax.q_spearman_predictor < 0.05, tax.p < 0.05],
                                ["FDR-significant adjusted (domain)", "FDR-significant adjusted (within predictor)",
                                 "FDR-significant Spearman (domain)", "FDR-significant Spearman (within predictor)", "Nominal only"],
                                default="Not significant")
    alp = pd.DataFrame(rows_alpha)
    alp = add_fdr(alp, ("family",), "p", "q_domain")
    alp = add_fdr(alp, ("family", "predictor"), "p", "q_predictor")
    alp = add_fdr(alp, ("family",), "rho_p", "q_spearman_domain")
    alp = add_fdr(alp, ("family", "predictor"), "rho_p", "q_spearman_predictor")
    alp["evidence"] = np.select([alp.q_domain < 0.05, alp.q_predictor < 0.05,
                                 alp.q_spearman_domain < 0.05, alp.q_spearman_predictor < 0.05, alp.p < 0.05],
                                ["FDR-significant adjusted (domain)", "FDR-significant adjusted (within predictor)",
                                 "FDR-significant Spearman (domain)", "FDR-significant Spearman (within predictor)", "Nominal only"],
                                default="Not significant")
    return tax.sort_values(["q_domain", "q_predictor", "p"]), alp.sort_values(["q_domain", "p"]), clin, Y, A


def paired_changes(w, alpha, primary_codes):
    rows = []
    key = [(HOMA, "Metabolic"), (PA, "Physical activity")]
    key += [(x, "Diet") for x in DIET]
    key += [(x, "Immune") for x in IMMUNE]
    for group_value, group_label in [(0, "Placebo"), (1, "Synbiotic")]:
        sub = w[w["Group_0"] == group_value]
        for var, domain in key:
            a = pd.to_numeric(sub[f"{var}_0"], errors="coerce")
            z = pd.to_numeric(sub[f"{var}_1"], errors="coerce")
            pair = pd.DataFrame({"a": a, "z": z}).dropna()
            delta = pair.z - pair.a
            if len(pair) < 5:
                continue
            try:
                wp = float(stats.wilcoxon(pair.z, pair.a, zero_method="wilcox").pvalue) if np.any(delta != 0) else 1.0
            except ValueError:
                wp = np.nan
            tp = float(stats.ttest_rel(pair.z, pair.a, nan_policy="omit").pvalue)
            rows.append({"analysis": "Paired 0-3 months", "arm": group_label, "domain": domain, "outcome": var,
                         "n": len(pair), "baseline_mean": pair.a.mean(), "month3_mean": pair.z.mean(),
                         "baseline_median": pair.a.median(), "month3_median": pair.z.median(),
                         "mean_change": delta.mean(), "median_change": delta.median(),
                         "p_wilcoxon": wp, "p_paired_t": tp})

    # Alpha diversity paired changes in the primary microbiome cohort.
    for group_value, group_label in [(0, "Placebo"), (1, "Synbiotic")]:
        codes = [c for c in primary_codes if int(w.loc[c, "Group_0"]) == group_value]
        for metric in alpha.columns:
            a = pd.Series([alpha.loc[str(w.loc[c, "Participant ID_0"]), metric] for c in codes], index=codes)
            z = pd.Series([alpha.loc[str(w.loc[c, "Participant ID_1"]), metric] for c in codes], index=codes)
            delta = z - a
            wp = float(stats.wilcoxon(z, a).pvalue) if np.any(delta != 0) else 1.0
            tp = float(stats.ttest_rel(z, a).pvalue)
            rows.append({"analysis": "Paired 0-3 months", "arm": group_label, "domain": "Alpha diversity", "outcome": metric,
                         "n": len(codes), "baseline_mean": a.mean(), "month3_mean": z.mean(),
                         "baseline_median": a.median(), "month3_median": z.median(),
                         "mean_change": delta.mean(), "median_change": delta.median(),
                         "p_wilcoxon": wp, "p_paired_t": tp})
    out = pd.DataFrame(rows)
    out = add_fdr(out, ("arm", "domain"), "p_wilcoxon", "q_wilcoxon")
    out["evidence"] = np.select([out.q_wilcoxon < 0.05, out.p_wilcoxon < 0.05], ["FDR-significant", "Nominal only"], default="Not significant")
    return out.sort_values(["q_wilcoxon", "p_wilcoxon"])


def longitudinal_nonmicro(w):
    rows = []
    data = w.copy()
    age, sex, bmi0, group, fhd = data["Age_0"], data["Sex_F_0"], data["BMI_0"], data["Group_0"], data["FHD_binary_0"]
    core = [("Group", group, True), ("Age", age, False), ("Sex_F", sex, True), ("BMI_0", bmi0, False)]

    def fitrow(family, pred_name, pred, outcome, y1, y0, pred_binary=False, extra=None, note=""):
        specs = [(pred_name, pred, pred_binary), ("baseline_outcome", y0, False)] + (extra if extra is not None else core)
        fit = fit_hc3(y1, data, pred_name, specs)
        if fit:
            rho, rho_p = spearman_pair(pred, y1 - y0)
            rows.append({"analysis": "Longitudinal 0-3 months", "family": family, "predictor": pred_name, "outcome": outcome,
                         "rho_with_change": rho, "rho_p": rho_p, **fit, "model": note})

    outcomes = [(HOMA, "log_HOMA"), (PA, "log_MET")] + [(x, x) for x in DIET] + [(x, f"log_{x}") for x in IMMUNE]
    # Intervention effects (ANCOVA).
    for label, transformed in outcomes:
        extra = [("Age", age, False), ("Sex_F", sex, True)] if label == "BMI" else [("Age", age, False), ("Sex_F", sex, True), ("BMI_0", bmi0, False)]
        fitrow("Treatment effect", "Synbiotic vs placebo", group, label, data[f"{transformed}_1"], data[f"{transformed}_0"], True, extra,
               "month-3 outcome ~ baseline outcome + treatment + age + sex + baseline BMI")

    # Lifestyle changes related to immune changes.
    for y in IMMUNE:
        fitrow("Change PA-immune", "Change in log MET", data["d_log_MET"], y,
               data[f"log_{y}_1"], data[f"log_{y}_0"], False, core,
               "month-3 log1p(cytokine) ~ baseline cytokine + change log1p(MET) + treatment + age + sex + baseline BMI")
        for x in DIET:
            fitrow("Change diet-immune", f"Change in {x}", data[f"d_{x}"], y,
                   data[f"log_{y}_1"], data[f"log_{y}_0"], False, core,
                   "month-3 log1p(cytokine) ~ baseline cytokine + diet-score change + treatment + age + sex + baseline BMI")

    # Lifestyle and immune changes related to HOMA-IR change.
    fitrow("Change lifestyle-HOMA", "Change in log MET", data["d_log_MET"], HOMA,
           data["log_HOMA_1"], data["log_HOMA_0"], False, core)
    for x in DIET:
        fitrow("Change lifestyle-HOMA", f"Change in {x}", data[f"d_{x}"], HOMA,
               data["log_HOMA_1"], data["log_HOMA_0"], False, core)
    for x in IMMUNE:
        fitrow("Change immune-HOMA", f"Change in {x}", data[f"d_log_{x}"], HOMA,
               data["log_HOMA_1"], data["log_HOMA_0"], False, core)

    # Family-history association with change.
    for label, transformed in outcomes:
        fitrow("FHD-change", "Family history (yes vs no)", fhd, label,
               data[f"{transformed}_1"], data[f"{transformed}_0"], True, core,
               "month-3 outcome ~ baseline outcome + family history + treatment + age + sex + baseline BMI; code 2 excluded")

    # Treatment x family-history moderation.
    inter = group * fhd
    for label, transformed in outcomes:
        specs = [("Treatment x FHD", inter, True), ("Group", group, True), ("FHD", fhd, True),
                 ("baseline_outcome", data[f"{transformed}_0"], False), ("Age", age, False),
                 ("Sex_F", sex, True), ("BMI_0", bmi0, False)]
        fit = fit_hc3(data[f"{transformed}_1"], data, "Treatment x FHD", specs)
        if fit:
            rows.append({"analysis": "Longitudinal 0-3 months", "family": "Treatment x FHD", "predictor": "Treatment x FHD",
                         "outcome": label, "rho_with_change": np.nan, "rho_p": np.nan, **fit,
                         "model": "month-3 outcome ~ baseline + treatment * family history + age + sex + baseline BMI; code 2 excluded"})

    out = pd.DataFrame(rows)
    out = add_fdr(out, ("family",), "p", "q_family")
    out = add_fdr(out, ("family",), "rho_p", "q_spearman")
    out["evidence"] = np.select(
        [out.q_family < 0.05, out.q_spearman < 0.05, out.p < 0.05],
        ["FDR-significant adjusted", "FDR-significant Spearman only", "Nominal only"],
        default="Not significant")
    return out.sort_values(["q_family", "p"])


def longitudinal_microbiome(w, clr, alpha, taxa, primary_codes):
    codes = primary_codes
    data = w.loc[codes].copy()
    s0 = [str(data.loc[c, "Participant ID_0"]) for c in codes]
    s1 = [str(data.loc[c, "Participant ID_1"]) for c in codes]
    y0 = clr.loc[s0, taxa].copy(); y0.index = codes
    y1 = clr.loc[s1, taxa].copy(); y1.index = codes
    dy = y1 - y0
    a0 = alpha.loc[s0].copy(); a0.index = codes
    a1 = alpha.loc[s1].copy(); a1.index = codes
    da = a1 - a0
    age, sex, bmi0, group, fhd = data["Age_0"], data["Sex_F_0"], data["BMI_0"], data["Group_0"], data["FHD_binary_0"]
    core = [("Group", group, True), ("Age", age, False), ("Sex_F", sex, True), ("BMI_0", bmi0, False)]
    tax_rows, alpha_rows, within_rows = [], [], []

    predictors = [("Change PA-taxa", "Change in log MET", data["d_log_MET"], False)]
    predictors += [("Change diet-taxa", f"Change in {x}", data[f"d_{x}"], False) for x in DIET]
    predictors += [("Change immune-taxa", f"Change in {x}", data[f"d_log_{x}"], False) for x in IMMUNE]
    predictors += [("Change HOMA-taxa", "Change in log HOMA-IR", data["d_log_HOMA"], False)]
    predictors += [("FHD-change taxa", "Family history (yes vs no)", fhd, True)]
    predictors += [("Treatment effect taxa", "Synbiotic vs placebo", group, True)]

    for family, pred_name, pred, binary in predictors:
        for taxon in taxa:
            specs = [(pred_name, pred, binary), ("baseline_taxon", y0[taxon], False)]
            if family == "Treatment effect taxa":
                specs += [("Age", age, False), ("Sex_F", sex, True), ("BMI_0", bmi0, False)]
            else:
                specs += core
            fit = fit_hc3(y1[taxon], data, pred_name, specs)
            if fit:
                rho, rho_p = spearman_pair(pred, dy[taxon])
                tax_rows.append({"analysis": "Longitudinal 0-3 months", "family": family, "predictor": pred_name,
                                 "taxon": taxon, "rho_with_change": rho, "rho_p": rho_p, **fit})
        for metric in alpha.columns:
            specs = [(pred_name, pred, binary), ("baseline_alpha", a0[metric], False)]
            if family == "Treatment effect taxa":
                specs += [("Age", age, False), ("Sex_F", sex, True), ("BMI_0", bmi0, False)]
            else:
                specs += core
            fit = fit_hc3(a1[metric], data, pred_name, specs)
            if fit:
                rho, rho_p = spearman_pair(pred, da[metric])
                alpha_rows.append({"analysis": "Longitudinal 0-3 months", "family": family.replace("taxa", "alpha"),
                                   "predictor": pred_name, "outcome": metric, "rho_with_change": rho, "rho_p": rho_p, **fit})

    # Baseline taxa as predictors of HOMA-IR response.
    for taxon in taxa:
        specs = [("Baseline taxon CLR", y0[taxon], False), ("baseline_HOMA", data["log_HOMA_0"], False)] + core
        fit = fit_hc3(data["log_HOMA_1"], data, "Baseline taxon CLR", specs)
        if fit:
            rho, rho_p = spearman_pair(y0[taxon], data["d_log_HOMA"])
            tax_rows.append({"analysis": "Longitudinal 0-3 months", "family": "Baseline taxa-HOMA response",
                             "predictor": "Baseline taxon CLR", "taxon": taxon,
                             "rho_with_change": rho, "rho_p": rho_p, **fit})

    # Treatment x family-history moderation.
    inter = group * fhd
    for taxon in taxa:
        specs = [("Treatment x FHD", inter, True), ("Group", group, True), ("FHD", fhd, True),
                 ("baseline_taxon", y0[taxon], False), ("Age", age, False), ("Sex_F", sex, True), ("BMI_0", bmi0, False)]
        fit = fit_hc3(y1[taxon], data, "Treatment x FHD", specs)
        if fit:
            tax_rows.append({"analysis": "Longitudinal 0-3 months", "family": "Treatment x FHD taxa",
                             "predictor": "Treatment x FHD", "taxon": taxon,
                             "rho_with_change": np.nan, "rho_p": np.nan, **fit})
    for metric in alpha.columns:
        specs = [("Treatment x FHD", inter, True), ("Group", group, True), ("FHD", fhd, True),
                 ("baseline_alpha", a0[metric], False), ("Age", age, False), ("Sex_F", sex, True), ("BMI_0", bmi0, False)]
        fit = fit_hc3(a1[metric], data, "Treatment x FHD", specs)
        if fit:
            alpha_rows.append({"analysis": "Longitudinal 0-3 months", "family": "Treatment x FHD alpha",
                               "predictor": "Treatment x FHD", "outcome": metric,
                               "rho_with_change": np.nan, "rho_p": np.nan, **fit})

    # Paired within-arm CLR shifts.
    for g, arm in [(0, "Placebo"), (1, "Synbiotic")]:
        idx = data.index[group == g]
        for taxon in taxa:
            delta = dy.loc[idx, taxon]
            tt = stats.ttest_1samp(delta, 0, nan_policy="omit")
            try:
                wp = stats.wilcoxon(delta).pvalue if np.any(delta != 0) else 1.0
            except ValueError:
                wp = np.nan
            within_rows.append({"arm": arm, "taxon": taxon, "n": len(delta), "mean_delta_CLR": delta.mean(),
                                "median_delta_CLR": delta.median(), "p_paired_t": float(tt.pvalue), "p_wilcoxon": float(wp)})

    tax = pd.DataFrame(tax_rows)
    tax = add_fdr(tax, ("family",), "p", "q_domain")
    tax = add_fdr(tax, ("family", "predictor"), "p", "q_predictor")
    tax = add_fdr(tax, ("family",), "rho_p", "q_spearman_domain")
    tax = add_fdr(tax, ("family", "predictor"), "rho_p", "q_spearman_predictor")
    tax["evidence"] = np.select([tax.q_domain < 0.05, tax.q_predictor < 0.05,
                                 tax.q_spearman_domain < 0.05, tax.q_spearman_predictor < 0.05, tax.p < 0.05],
                                ["FDR-significant adjusted (domain)", "FDR-significant adjusted (within predictor)",
                                 "FDR-significant Spearman (domain)", "FDR-significant Spearman (within predictor)", "Nominal only"],
                                default="Not significant")
    alp = pd.DataFrame(alpha_rows)
    alp = add_fdr(alp, ("family",), "p", "q_domain")
    alp = add_fdr(alp, ("family", "predictor"), "p", "q_predictor")
    alp = add_fdr(alp, ("family",), "rho_p", "q_spearman_domain")
    alp = add_fdr(alp, ("family", "predictor"), "rho_p", "q_spearman_predictor")
    alp["evidence"] = np.select([alp.q_domain < 0.05, alp.q_predictor < 0.05,
                                 alp.q_spearman_domain < 0.05, alp.q_spearman_predictor < 0.05, alp.p < 0.05],
                                ["FDR-significant adjusted (domain)", "FDR-significant adjusted (within predictor)",
                                 "FDR-significant Spearman (domain)", "FDR-significant Spearman (within predictor)", "Nominal only"],
                                default="Not significant")
    within = pd.DataFrame(within_rows)
    within = add_fdr(within, ("arm",), "p_paired_t", "q_paired_t")
    within = add_fdr(within, ("arm",), "p_wilcoxon", "q_wilcoxon")
    within["evidence"] = np.select([within.q_paired_t < 0.05, within.p_paired_t < 0.05], ["FDR-significant", "Nominal only"], default="Not significant")
    return (tax.sort_values(["q_domain", "q_predictor", "p"]),
            alp.sort_values(["q_domain", "p"]), within.sort_values(["q_paired_t", "p_paired_t"]),
            data, y0, y1, dy, a0, a1, da)


def partial_permanova(Y, reduced_cols, effect_cols, permutations=N_PERM, seed=20260826):
    Y = np.asarray(Y, dtype=float)
    red = np.asarray(reduced_cols, dtype=float)
    eff = np.asarray(effect_cols, dtype=float)
    if red.ndim == 1:
        red = red[:, None]
    if eff.ndim == 1:
        eff = eff[:, None]
    finite = np.isfinite(Y).all(axis=1) & np.isfinite(red).all(axis=1) & np.isfinite(eff).all(axis=1)
    Y, red, eff = Y[finite], red[finite], eff[finite]
    if len(Y) < red.shape[1] + eff.shape[1] + 6:
        return None
    Xr = np.column_stack([np.ones(len(Y)), red])
    Xf = np.column_stack([Xr, eff])
    rr, rf = np.linalg.matrix_rank(Xr), np.linalg.matrix_rank(Xf)
    df1, df2 = rf - rr, len(Y) - rf
    if df1 <= 0 or df2 <= 0:
        return None

    Qr = np.linalg.qr(Xr, mode="reduced")[0]
    Qf = np.linalg.qr(Xf, mode="reduced")[0]
    fit_r = Qr @ (Qr.T @ Y)
    resid_r = Y - fit_r

    def sse(Q, Z):
        R = Z - Q @ (Q.T @ Z)
        return float(np.sum(R * R))

    sser = sse(Qr, Y)
    ssef = sse(Qf, Y)
    numer = max(sser - ssef, 0)
    F = (numer / df1) / (ssef / df2) if ssef > 0 else np.inf
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(permutations):
        Z = fit_r + resid_r[rng.permutation(len(Y)), :]
        pr = sse(Qr, Z)
        pf = sse(Qf, Z)
        pF = ((max(pr - pf, 0) / df1) / (pf / df2)) if pf > 0 else np.inf
        ge += pF >= F - 1e-12
    return {"n": int(len(Y)), "partial_R2": float(numer / sser) if sser > 0 else np.nan,
            "pseudo_F": float(F), "df_effect": int(df1), "df_residual": int(df2),
            "p": float((ge + 1) / (permutations + 1)), "permutations": permutations}


def global_microbiome_tests(bclin, Ybase, ldata, dy):
    rows = []

    def zcol(s):
        return safe_z(pd.to_numeric(s, errors="coerce").to_numpy())

    # Baseline partial Aitchison-space tests.
    core = np.column_stack([zcol(bclin.Age), bclin.Sex_F.to_numpy(), zcol(bclin.BMI)])
    tests = [("Baseline PA", "Physical activity (log MET-min/wk)", zcol(bclin.log_MET), core)]
    for x in DIET:
        tests.append(("Baseline diet", x, zcol(bclin[x]), np.column_stack([core, zcol(bclin.log_MET)])))
    for x in IMMUNE:
        tests.append(("Baseline immune", x, zcol(bclin[f"log_{x}"]), np.column_stack([core, zcol(bclin.log_MET)])))
    tests.append(("Baseline FHD", "Family history (yes vs no)", bclin.FHD_binary.to_numpy(), core))
    tests.append(("Baseline HOMA", HOMA, zcol(bclin.log_HOMA), np.column_stack([core, zcol(bclin.log_MET)])))
    for i, (family, pred, effect, reduced) in enumerate(tests):
        r = partial_permanova(Ybase.to_numpy(), reduced, effect, seed=20260826 + i)
        if r:
            rows.append({"analysis": "Baseline Aitchison", "family": family, "predictor": pred, **r})

    # Longitudinal tests use CLR change as multivariate response.
    core_l = np.column_stack([ldata.Group_0.to_numpy(), zcol(ldata.Age_0), ldata.Sex_F_0.to_numpy(), zcol(ldata.BMI_0)])
    tests_l = [("Treatment effect", "Synbiotic vs placebo", ldata.Group_0.to_numpy(),
                np.column_stack([zcol(ldata.Age_0), ldata.Sex_F_0.to_numpy(), zcol(ldata.BMI_0)]))]
    tests_l.append(("Change PA", "Change in log MET", zcol(ldata.d_log_MET), core_l))
    for x in DIET:
        tests_l.append(("Change diet", f"Change in {x}", zcol(ldata[f"d_{x}"]), core_l))
    for x in IMMUNE:
        tests_l.append(("Change immune", f"Change in {x}", zcol(ldata[f"d_log_{x}"]), core_l))
    tests_l.append(("Change HOMA", "Change in log HOMA-IR", zcol(ldata.d_log_HOMA), core_l))
    tests_l.append(("FHD-change", "Family history (yes vs no)", ldata.FHD_binary_0.to_numpy(), core_l))
    fhd = ldata.FHD_binary_0.to_numpy()
    grp = ldata.Group_0.to_numpy()
    interaction = grp * fhd
    reduced_int = np.column_stack([grp, fhd, zcol(ldata.Age_0), ldata.Sex_F_0.to_numpy(), zcol(ldata.BMI_0)])
    tests_l.append(("Treatment x FHD", "Treatment x family history", interaction, reduced_int))
    for i, (family, pred, effect, reduced) in enumerate(tests_l):
        r = partial_permanova(dy.to_numpy(), reduced, effect, seed=20270000 + i)
        if r:
            rows.append({"analysis": "Longitudinal Aitchison change", "family": family, "predictor": pred, **r})

    out = pd.DataFrame(rows)
    out = add_fdr(out, ("analysis",), "p", "q_analysis")
    out = add_fdr(out, ("analysis", "family"), "p", "q_family")
    out["evidence"] = np.select([out.q_analysis < 0.05, out.q_family < 0.05, out.p < 0.05],
                                ["FDR-significant (analysis)", "FDR-significant (family)", "Nominal only"],
                                default="Not significant")
    return out.sort_values(["q_analysis", "p"])


def sensitivity_taxa(w, clr, taxa, sensitivity_codes):
    """Sensitivity for baseline PA-taxa including all paired samples, including antibiotic-exposed IDs."""
    codes = sensitivity_codes
    b = w.loc[codes]
    sample_ids = [str(b.loc[c, "Participant ID_0"]) for c in codes]
    Y = clr.loc[sample_ids, taxa].copy(); Y.index = codes
    rows = []
    covs = [("Age", b.Age_0, False), ("Sex_F", b.Sex_F_0, True), ("BMI", b.BMI_0, False)]
    for taxon in taxa:
        fit = fit_hc3(Y[taxon], b, "Physical activity (log MET-min/wk)",
                      [("Physical activity (log MET-min/wk)", b.log_MET_0, False)] + covs)
        if fit:
            rows.append({"analysis": "Sensitivity including antibiotic-exposed paired samples", "family": "PA-taxa",
                         "predictor": "Physical activity (log MET-min/wk)", "taxon": taxon, **fit})
    out = pd.DataFrame(rows)
    out["q_domain"] = bh_adjust(out.p)
    return out.sort_values(["q_domain", "p"])


def build_triangles(base_tax, long_tax):
    rows = []
    # Same-taxon two-link patterns. These are coherence screens, not mediation tests.
    def screen(df, diet_family, immune_family, stage):
        dd = df[df.family == diet_family]
        ii = df[df.family == immune_family]
        for taxon in sorted(set(dd.taxon) & set(ii.taxon)):
            dsub = dd[dd.taxon == taxon]
            isub = ii[ii.taxon == taxon]
            for _, dr in dsub.iterrows():
                for _, ir in isub.iterrows():
                    if dr.q_predictor < 0.10 and ir.q_predictor < 0.10:
                        rows.append({"stage": stage, "diet_link": dr.predictor, "taxon": taxon,
                                     "immune_link": ir.predictor, "diet_beta": dr.beta, "diet_q": dr.q_predictor,
                                     "immune_beta": ir.beta, "immune_q": ir.q_predictor,
                                     "evidence": "Two-link FDR q<0.10 coherence; not causal mediation"})
    screen(base_tax, "Diet-taxa", "Immune-taxa", "Baseline")
    screen(long_tax, "Change diet-taxa", "Change immune-taxa", "Longitudinal change")
    columns = ["stage", "diet_link", "taxon", "immune_link", "diet_beta", "diet_q", "immune_beta", "immune_q", "evidence"]
    return pd.DataFrame(rows, columns=columns)


def gather_top_hits(*tables):
    rows = []
    for name, df, qcol, targetcol in tables:
        if df is None or len(df) == 0:
            continue
        for _, r in df.iterrows():
            q = r.get(qcol, np.nan)
            p = r.get("p", r.get("p_wilcoxon", r.get("p_paired_t", np.nan)))
            sqcol = "q_spearman_domain" if "q_spearman_domain" in df.columns else ("q_spearman" if "q_spearman" in df.columns else None)
            sq = r.get(sqcol, np.nan) if sqcol else np.nan
            sp = r.get("rho_p", np.nan)
            adjusted_robust = np.isfinite(q) and q < 0.05
            spearman_robust = np.isfinite(sq) and sq < 0.05
            exploratory = np.isfinite(p) and p < 0.01
            if adjusted_robust or spearman_robust or exploratory:
                label = r.get("outcome", r.get("taxon", r.get("predictor", "")))
                predictor = r.get("predictor", r.get("arm", ""))
                if spearman_robust and not adjusted_robust:
                    effect = r.get("rho", r.get("rho_with_change", np.nan))
                    final_p, final_q = sp, sq
                    status = "FDR-significant Spearman only"
                else:
                    effect = r.get("beta", r.get("median_change", r.get("partial_R2", r.get("mean_delta_CLR", np.nan))))
                    final_p, final_q = p, q
                    status = "FDR-significant adjusted" if adjusted_robust else "Exploratory p<0.01"
                rows.append({"source_table": name, "family": r.get("family", r.get("domain", "")),
                             "taxonomic_level": r.get("taxonomic_level", ""),
                             "predictor": predictor, "outcome_or_taxon": label, "effect": effect,
                             "p": final_p, "q": final_q, "status": status})
    if not rows:
        return pd.DataFrame(columns=["source_table", "family", "taxonomic_level", "predictor", "outcome_or_taxon", "effect", "p", "q", "status"])
    out = pd.DataFrame(rows).drop_duplicates()
    out["sort_q"] = out.q.fillna(1)
    status_order = {"FDR-significant adjusted": 0, "FDR-significant Spearman only": 1, "Exploratory p<0.01": 2}
    out["sort_status"] = out.status.map(status_order).fillna(9)
    out = out.sort_values(["sort_status", "sort_q", "p"]).drop(columns=["sort_q", "sort_status"])
    return out


def main():
    d = clean_clinical()
    b, e, w = make_wide(d)
    (counts, rel, clr, alpha, taxa, primary_codes, sensitivity_codes, prevalence,
     genus_clr, genera, genus_prevalence) = microbiome_objects(d, b, e, w)

    base_non = baseline_nonmicro(b)
    base_tax_sp, base_alpha, bclin, Ybase, Abase = baseline_microbiome(b, w, clr, alpha, taxa, primary_codes)
    base_tax_sp["taxonomic_level"] = "Species"
    base_tax_ge, _base_alpha_ge, bclin_ge, Ybase_ge, _ = baseline_microbiome(b, w, genus_clr, alpha, genera, primary_codes)
    base_tax_ge["taxonomic_level"] = "Genus"
    base_tax = pd.concat([base_tax_sp, base_tax_ge], ignore_index=True)
    paired = paired_changes(w, alpha, primary_codes)
    long_non = longitudinal_nonmicro(w)
    long_tax_sp, long_alpha, within_tax_sp, ldata, y0, y1, dy, a0, a1, da = longitudinal_microbiome(w, clr, alpha, taxa, primary_codes)
    long_tax_sp["taxonomic_level"] = "Species"
    within_tax_sp["taxonomic_level"] = "Species"
    long_tax_ge, _long_alpha_ge, within_tax_ge, ldata_ge, gy0, gy1, gdy, *_ = longitudinal_microbiome(w, genus_clr, alpha, genera, primary_codes)
    long_tax_ge["taxonomic_level"] = "Genus"
    within_tax_ge["taxonomic_level"] = "Genus"
    long_tax = pd.concat([long_tax_sp, long_tax_ge], ignore_index=True)
    within_tax = pd.concat([within_tax_sp, within_tax_ge], ignore_index=True)
    permanova_sp = global_microbiome_tests(bclin, Ybase, ldata, dy)
    permanova_sp["taxonomic_level"] = "Species"
    permanova_ge = global_microbiome_tests(bclin_ge, Ybase_ge, ldata_ge, gdy)
    permanova_ge["taxonomic_level"] = "Genus"
    permanova = pd.concat([permanova_sp, permanova_ge], ignore_index=True)
    sensitivity_sp = sensitivity_taxa(w, clr, taxa, sensitivity_codes)
    sensitivity_sp["taxonomic_level"] = "Species"
    sensitivity_ge = sensitivity_taxa(w, genus_clr, genera, sensitivity_codes)
    sensitivity_ge["taxonomic_level"] = "Genus"
    sensitivity = pd.concat([sensitivity_sp, sensitivity_ge], ignore_index=True)
    triangles = build_triangles(base_tax, long_tax)

    top = gather_top_hits(
        ("Baseline cross-domain", base_non, "q_family", "outcome"),
        ("Baseline taxa", base_tax, "q_domain", "taxon"),
        ("Baseline alpha", base_alpha, "q_domain", "outcome"),
        ("Paired changes", paired, "q_wilcoxon", "outcome"),
        ("Longitudinal cross-domain", long_non, "q_family", "outcome"),
        ("Longitudinal taxa", long_tax, "q_domain", "taxon"),
        ("Longitudinal alpha", long_alpha, "q_domain", "outcome"),
        ("Within-arm taxa", within_tax.rename(columns={"p_paired_t": "p"}), "q_paired_t", "taxon"),
        ("Global microbiome", permanova, "q_analysis", "predictor"),
    )

    # Analysis-ready participant-level data for audit (no raw ASV matrix duplicated).
    analysis_ready = d[["Participant ID", "Code #", "timeline", "Intervention_label", "Gender", "Age", "BMI",
                        "family history of diabetes", "FHD_binary", HOMA, PA, *DIET, *IMMUNE]].copy()
    analysis_ready["microbiome_primary_eligible"] = analysis_ready["Code #"].isin(primary_codes)
    analysis_ready["family_history_note"] = np.where(analysis_ready["family history of diabetes"] == 2,
                                                       "Code 2 excluded from binary family-history models", "")

    qc = pd.DataFrame([
        ["Clinical valid records", len(d), "Rows with timeline 0 or 1"],
        ["Clinical paired participants", len(w), "Complete baseline and month-3 clinical records"],
        ["Microbiome paired participants before antibiotic exclusion", len(sensitivity_codes), "Both ASV visits plus clinical data"],
        ["Microbiome primary paired participants", len(primary_codes), "Excludes antibiotic-use IDs 150, 168, 271 when present"],
        ["Filtered taxa", len(taxa), "Baseline prevalence >=20% in primary microbiome cohort"],
        ["Filtered genera", len(genera), "Genus aggregation; baseline prevalence >=20% in primary microbiome cohort"],
        ["Family-history yes/no participants", int(b.FHD_binary.notna().sum()), "Code 2 excluded from binary family-history models"],
        ["Family-history code 2 participants", int((b["family history of diabetes"] == 2).sum()), "Meaning not documented in supplied file"],
        ["IL-6 decimal-comma values repaired", 1, "String 7,3 parsed as 7.3"],
        ["Microbiome permutations", N_PERM, "Freedman-Lane partial multivariate tests"],
    ], columns=["item", "value", "definition"])

    outputs = {
        "qc_summary.csv": qc,
        "analysis_ready_clinical.csv": analysis_ready,
        "top_findings.csv": top,
        "baseline_cross_domain.csv": base_non,
        "baseline_taxa.csv": base_tax,
        "baseline_alpha.csv": base_alpha,
        "paired_changes.csv": paired,
        "longitudinal_cross_domain.csv": long_non,
        "longitudinal_taxa.csv": long_tax,
        "longitudinal_alpha.csv": long_alpha,
        "within_arm_taxa.csv": within_tax,
        "global_microbiome.csv": permanova,
        "sensitivity_PA_taxa.csv": sensitivity,
        "diet_taxa_immune_triangles.csv": triangles,
        "taxa_prevalence.csv": pd.concat([
            pd.DataFrame({"taxonomic_level": "Species", "taxon": prevalence.index, "baseline_prevalence": prevalence.values,
                          "included_prevalence_20pct": prevalence.index.isin(taxa)}),
            pd.DataFrame({"taxonomic_level": "Genus", "taxon": genus_prevalence.index, "baseline_prevalence": genus_prevalence.values,
                          "included_prevalence_20pct": genus_prevalence.index.isin(genera)}),
        ], ignore_index=True),
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False)

    summary = {
        "clinical_participants": len(w),
        "microbiome_primary_participants": len(primary_codes),
        "microbiome_sensitivity_participants": len(sensitivity_codes),
        "species_after_filter": len(taxa),
        "genera_after_filter": len(genera),
        "robust_top_hits": int(top.status.astype(str).str.startswith("FDR-significant").sum()) if len(top) else 0,
        "exploratory_top_hits": int((top.status == "Exploratory p<0.01").sum()) if len(top) else 0,
        "primary_codes": primary_codes,
        "antibiotic_exclusions": sorted(ANTIBIOTIC_EXCLUSIONS),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Top findings")
    print(top.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
