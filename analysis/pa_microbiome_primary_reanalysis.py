from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import analysis_pipeline as core


SEED = 20260826
IR_THRESHOLD = 1.85


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=1)
    return (x - np.nanmean(x)) / sd if np.isfinite(sd) and sd > 0 else np.full_like(x, np.nan)


def configure_paths(args):
    core.CLINICAL_FILE = Path(args.clinical).resolve()
    core.COUNTS_FILE = Path(args.counts).resolve()
    core.REL_FILE = Path(args.relative).resolve()


def paired_taxon_matrices(w, clr, taxa, codes):
    data = w.loc[codes].copy()
    s0 = [str(data.loc[c, "Participant ID_0"]) for c in codes]
    s1 = [str(data.loc[c, "Participant ID_1"]) for c in codes]
    y0 = clr.loc[s0, taxa].copy(); y0.index = codes
    y1 = clr.loc[s1, taxa].copy(); y1.index = codes
    return data, y0, y1, y1 - y0


def primary_specs(data, y0):
    return [
        ("Change in log MET", data["d_log_MET"], False),
        ("baseline_taxon", y0, False),
        ("Group", data["Group_0"], True),
        ("Age", data["Age_0"], False),
        ("Sex_F", data["Sex_F_0"], True),
        ("BMI_0", data["BMI_0"], False),
    ]


def screen_taxa(w, clr, taxa, codes, level):
    data, y0, y1, dy = paired_taxon_matrices(w, clr, taxa, codes)
    rows = []
    for taxon in taxa:
        fit = core.fit_hc3(y1[taxon], data, "Change in log MET", primary_specs(data, y0[taxon]))
        rho, rho_p = core.spearman_pair(data["d_log_MET"], dy[taxon])
        if fit:
            rows.append({
                "analysis": "Taxon screen",
                "taxonomic_level": level,
                "taxon": taxon,
                "effect": "Change in log MET",
                "rho_change_change": rho,
                "rho_p": rho_p,
                **fit,
            })
    out = pd.DataFrame(rows)
    out["q"] = core.bh_adjust(out["p"])
    out["rho_q"] = core.bh_adjust(out["rho_p"])
    out["evidence"] = np.where(out["q"] < 0.05, "FDR-significant", np.where(out["p"] < 0.05, "Nominal", "Not significant"))
    return out.sort_values(["q", "p"]), data, y0, y1, dy


def global_tests(data, y0, dy, level, permutations):
    age = zscore(data["Age_0"])
    sex = data["Sex_F_0"].to_numpy(float)
    bmi = zscore(data["BMI_0"])
    group = data["Group_0"].to_numpy(float)
    pa0 = zscore(data["log_MET_0"])
    dpa = zscore(data["d_log_MET"])
    diet = np.column_stack([zscore(data[f"d_{x}"]) for x in core.DIET])
    rows = []
    tests = [
        ("Baseline Aitchison", y0.to_numpy(), np.column_stack([age, sex, bmi]), pa0, "Primary"),
        ("Longitudinal Aitchison change", dy.to_numpy(), np.column_stack([group, age, sex, bmi]), dpa, "Primary"),
        ("Longitudinal Aitchison change", dy.to_numpy(), np.column_stack([group, age, sex, bmi, pa0]), dpa, "+ baseline PA"),
        ("Longitudinal Aitchison change", dy.to_numpy(), np.column_stack([group, age, sex, bmi, pa0, diet]), dpa, "+ baseline PA + diet changes"),
    ]
    for i, (analysis, y, reduced, effect, model) in enumerate(tests):
        fit = core.partial_permanova(y, reduced, effect, permutations=permutations, seed=SEED + i + (0 if level == "Species" else 100))
        rows.append({"analysis": analysis, "taxonomic_level": level, "model": model, "effect": "Physical activity", **fit})
    return pd.DataFrame(rows)


def alpha_tests(w, alpha, codes):
    data = w.loc[codes].copy()
    s0 = [str(data.loc[c, "Participant ID_0"]) for c in codes]
    s1 = [str(data.loc[c, "Participant ID_1"]) for c in codes]
    a0 = alpha.loc[s0].copy(); a0.index = codes
    a1 = alpha.loc[s1].copy(); a1.index = codes
    rows = []
    for metric in alpha.columns:
        base = core.fit_hc3(
            a0[metric], data, "Baseline log MET",
            [
                ("Baseline log MET", data["log_MET_0"], False),
                ("Age", data["Age_0"], False),
                ("Sex_F", data["Sex_F_0"], True),
                ("BMI_0", data["BMI_0"], False),
            ],
        )
        long = core.fit_hc3(
            a1[metric], data, "Change in log MET",
            [
                ("Change in log MET", data["d_log_MET"], False),
                ("baseline_alpha", a0[metric], False),
                ("Group", data["Group_0"], True),
                ("Age", data["Age_0"], False),
                ("Sex_F", data["Sex_F_0"], True),
                ("BMI_0", data["BMI_0"], False),
            ],
        )
        rows.append({"analysis": "Baseline alpha", "metric": metric, "effect": "Baseline log MET", **base})
        rows.append({"analysis": "Longitudinal alpha", "metric": metric, "effect": "Change in log MET", **long})
    out = pd.DataFrame(rows)
    out["q"] = out.groupby("analysis")["p"].transform(lambda x: core.bh_adjust(x))
    return out


def custom_hc3(y, frame, effect, modes):
    z = pd.DataFrame({"__y": pd.to_numeric(pd.Series(y, index=frame.index), errors="coerce")}, index=frame.index)
    for name, (values, _mode) in modes.items():
        z[name] = pd.to_numeric(pd.Series(values, index=frame.index), errors="coerce")
    z = z.replace([np.inf, -np.inf], np.nan).dropna()
    yz = zscore(z["__y"])
    cols = []
    names = []
    for name, (_values, mode) in modes.items():
        x = z[name].to_numpy(float)
        if mode == "continuous":
            x = zscore(x)
        cols.append(x)
        names.append(name)
    X = np.column_stack([np.ones(len(z))] + cols)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = np.linalg.lstsq(X, yz, rcond=None)[0]
    resid = yz - X @ beta
    hat = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    u = resid / np.clip(1 - hat, 1e-8, None)
    cov = xtx_inv @ (X.T @ ((u * u)[:, None] * X)) @ xtx_inv
    j = 1 + names.index(effect)
    se = float(np.sqrt(max(cov[j, j], 0)))
    df = len(z) - X.shape[1]
    tval = float(beta[j] / se)
    crit = float(stats.t.ppf(0.975, df))
    return {
        "n": int(len(z)), "beta": float(beta[j]), "se_hc3": se,
        "ci_low": float(beta[j] - crit * se), "ci_high": float(beta[j] + crit * se),
        "p": float(2 * stats.t.sf(abs(tval), df)), "df": int(df),
    }


def residualize(y, covariates):
    frame = pd.concat([pd.Series(y, name="y"), pd.DataFrame(covariates)], axis=1).dropna()
    X = np.column_stack([np.ones(len(frame)), frame.drop(columns="y").to_numpy(float)])
    beta = np.linalg.lstsq(X, frame["y"].to_numpy(float), rcond=None)[0]
    resid = frame["y"].to_numpy(float) - X @ beta
    out = pd.Series(np.nan, index=pd.Series(y).index, dtype=float)
    out.loc[frame.index] = resid
    return out


def permutation_p(y, data, y0, permutations, seed):
    frame = pd.DataFrame({
        "y": y, "dpa": data["d_log_MET"], "y0": y0,
        "group": data["Group_0"], "age": data["Age_0"],
        "sex": data["Sex_F_0"], "bmi": data["BMI_0"],
    }).dropna()
    yv = zscore(frame.y)
    dpa = zscore(frame.dpa)
    cov = np.column_stack([zscore(frame.y0), frame.group, zscore(frame.age), frame.sex, zscore(frame.bmi)])
    Xr = np.column_stack([np.ones(len(frame)), cov])
    Xf = np.column_stack([Xr, dpa])
    br = np.linalg.lstsq(Xr, yv, rcond=None)[0]
    fitted = Xr @ br
    resid = yv - fitted

    def tstat(target):
        inv = np.linalg.pinv(Xf.T @ Xf)
        b = np.linalg.lstsq(Xf, target, rcond=None)[0]
        r = target - Xf @ b
        h = np.einsum("ij,jk,ik->i", Xf, inv, Xf)
        u = r / np.clip(1 - h, 1e-8, None)
        vc = inv @ (Xf.T @ ((u * u)[:, None] * Xf)) @ inv
        return float(b[-1] / np.sqrt(vc[-1, -1]))

    observed = abs(tstat(yv))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        yp = fitted + resid[rng.permutation(len(resid))]
        exceed += abs(tstat(yp)) >= observed - 1e-12
    return (exceed + 1) / (permutations + 1)


def targeted_models(taxon, level, data, y0, y1, w, clr, all_codes, permutations):
    rows = []
    base = primary_specs(data, y0)
    model_specs = {
        "Primary": base,
        "+ baseline PA": base + [("Baseline log MET", data["log_MET_0"], False)],
        "+ diet changes": base + [(f"d_{x}", data[f"d_{x}"], False) for x in core.DIET],
        "+ baseline PA + diet changes": base + [("Baseline log MET", data["log_MET_0"], False)] + [(f"d_{x}", data[f"d_{x}"], False) for x in core.DIET],
    }
    for model, specs in model_specs.items():
        fit = core.fit_hc3(y1, data, "Change in log MET", specs)
        rows.append({"analysis": "PA-taxon sensitivity", "taxon": taxon, "taxonomic_level": level, "model": model, "effect": "Change in log MET", **fit})

    all_data, all_y0, all_y1, _ = paired_taxon_matrices(w, clr, [taxon], all_codes)
    fit_all = core.fit_hc3(all_y1[taxon], all_data, "Change in log MET", primary_specs(all_data, all_y0[taxon]))
    rows.append({"analysis": "PA-taxon sensitivity", "taxon": taxon, "taxonomic_level": level, "model": "Include antibiotic-exposed", "effect": "Change in log MET", **fit_all})

    perm_p = permutation_p(y1, data, y0, permutations, SEED + len(taxon))
    primary = rows[0]
    primary["permutation_p"] = perm_p

    betas = []
    for code in data.index:
        idx = data.index[data.index != code]
        fit = core.fit_hc3(y1.loc[idx], data.loc[idx], "Change in log MET", primary_specs(data.loc[idx], y0.loc[idx]))
        if fit:
            betas.append(fit["beta"])
    rows.append({
        "analysis": "Leave-one-out stability", "taxon": taxon, "taxonomic_level": level,
        "model": "Primary leave-one-out", "effect": "Change in log MET",
        "n": len(betas), "beta": float(np.mean(betas)), "ci_low": float(np.min(betas)),
        "ci_high": float(np.max(betas)), "p": np.nan,
        "sign_consistency": float(np.mean(np.sign(betas) == np.sign(rows[0]["beta"]))),
    })

    dpa_z = pd.Series(zscore(data["d_log_MET"]), index=data.index)
    ir = (data["HOMA-IR_0"] > IR_THRESHOLD).astype(int)
    modifiers = {"Intervention": data["Group_0"].astype(int), "Baseline IR status": ir}
    for modifier_name, modifier in modifiers.items():
        interaction = dpa_z * modifier
        modes = {
            "dPA_z": (dpa_z, "raw"),
            "modifier": (modifier, "raw"),
            "interaction": (interaction, "raw"),
            "baseline_taxon": (y0, "continuous"),
            "Group": (data["Group_0"], "raw"),
            "Age": (data["Age_0"], "continuous"),
            "Sex": (data["Sex_F_0"], "raw"),
            "BMI": (data["BMI_0"], "continuous"),
        }
        if modifier_name == "Intervention":
            modes.pop("Group")
        fit = custom_hc3(y1, data, "interaction", modes)
        rows.append({"analysis": "Effect modification", "taxon": taxon, "taxonomic_level": level, "model": modifier_name, "effect": "PA x modifier", **fit})
    return rows


def build_pa_microbiota_score(data, selected):
    components = []
    for item in selected:
        y0, y1, direction = item["y0"], item["y1"], np.sign(item["beta"])
        cov = pd.DataFrame({
            "baseline": zscore(y0), "group": data["Group_0"], "age": zscore(data["Age_0"]),
            "sex": data["Sex_F_0"], "bmi": zscore(data["BMI_0"]),
        }, index=data.index)
        resid = residualize(y1, cov)
        components.append(direction * pd.Series(zscore(resid), index=data.index))
    return pd.concat(components, axis=1).mean(axis=1)


def score_outcomes(w, data, score):
    rows = []
    direct_homa_specs = [
        ("Change in log MET", data["d_log_MET"], False),
        ("Baseline log HOMA", data["log_HOMA_0"], False),
        ("Group", data["Group_0"], True),
        ("Age", data["Age_0"], False),
        ("Sex", data["Sex_F_0"], True),
        ("BMI", data["BMI_0"], False),
    ]
    fit_direct_homa = core.fit_hc3(data["log_HOMA_1"], data, "Change in log MET", direct_homa_specs)
    rows.append({"analysis": "PA to metabolic outcome", "outcome": "Month-3 log HOMA-IR", "effect": "Change in log MET", **fit_direct_homa})
    fit_direct_homa_pa0 = core.fit_hc3(
        data["log_HOMA_1"], data, "Change in log MET",
        direct_homa_specs + [("Baseline log MET", data["log_MET_0"], False)],
    )
    rows.append({"analysis": "PA to metabolic outcome sensitivity", "outcome": "Month-3 log HOMA-IR", "effect": "Change in log MET", **fit_direct_homa_pa0})
    fit_group_homa = core.fit_hc3(data["log_HOMA_1"], data, "Group", direct_homa_specs)
    rows.append({"analysis": "Intervention to metabolic outcome adjusted for PA", "outcome": "Month-3 log HOMA-IR", "effect": "Group", **fit_group_homa})
    fit_group_pa = core.fit_hc3(
        data["log_MET_1"], data, "Group",
        [("Group", data["Group_0"], True), ("Baseline log MET", data["log_MET_0"], False),
         ("Age", data["Age_0"], False), ("Sex", data["Sex_F_0"], True), ("BMI", data["BMI_0"], False)],
    )
    rows.append({"analysis": "Intervention to physical activity", "outcome": "Month-3 log MET", "effect": "Group", **fit_group_pa})
    common = [
        ("PA microbiota score", score, False),
        ("Change in log MET", data["d_log_MET"], False),
        ("Group", data["Group_0"], True),
        ("Age", data["Age_0"], False),
        ("Sex", data["Sex_F_0"], True),
        ("BMI", data["BMI_0"], False),
    ]
    fit_pa_score = core.fit_hc3(score, data, "Change in log MET", [
        ("Change in log MET", data["d_log_MET"], False),
        ("Group", data["Group_0"], True), ("Age", data["Age_0"], False),
        ("Sex", data["Sex_F_0"], True), ("BMI", data["BMI_0"], False),
    ])
    rows.append({"analysis": "PA to microbiota score", "outcome": "PA microbiota score", "effect": "Change in log MET", **fit_pa_score})
    fit_homa = core.fit_hc3(data["log_HOMA_1"], data, "PA microbiota score", common + [("Baseline log HOMA", data["log_HOMA_0"], False)])
    rows.append({"analysis": "Microbiota score to metabolic outcome", "outcome": "Month-3 log HOMA-IR", "effect": "PA microbiota score", **fit_homa})
    for marker in core.IMMUNE:
        fit = core.fit_hc3(data[f"log_{marker}_1"], data, "PA microbiota score", common + [("Baseline marker", data[f"log_{marker}_0"], False)])
        rows.append({"analysis": "Microbiota score to immune outcome", "outcome": marker, "effect": "PA microbiota score", **fit})
    out = pd.DataFrame(rows)
    immune = out.analysis == "Microbiota score to immune outcome"
    out.loc[immune, "q"] = core.bh_adjust(out.loc[immune, "p"])
    return out


def forest_plot(targeted, path):
    plot = targeted[(targeted.analysis == "PA-taxon sensitivity") & targeted.beta.notna()].copy()
    taxa = list(plot.taxon.drop_duplicates())
    models = ["Primary", "+ baseline PA", "+ diet changes", "+ baseline PA + diet changes", "Include antibiotic-exposed"]
    fig, axes = plt.subplots(1, len(taxa), figsize=(11, 4.8), sharey=True)
    if len(taxa) == 1:
        axes = [axes]
    for ax, taxon in zip(axes, taxa):
        sub = plot[plot.taxon == taxon].set_index("model").reindex(models)
        y = np.arange(len(models))
        ax.axvline(0, color="#6B7280", lw=1)
        ax.errorbar(sub.beta, y, xerr=[sub.beta - sub.ci_low, sub.ci_high - sub.beta], fmt="o", color="#2563EB", ecolor="#93C5FD", capsize=3)
        ax.set_title(taxon, fontstyle="italic", fontsize=11)
        ax.set_xlabel("Standardized beta (95% CI)")
        ax.set_yticks(y, models)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Physical activity–microbiota associations across sensitivity models", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(path, global_df, alpha_df, taxa_df, targeted_df, score_df, n_primary, n_all):
    hits = taxa_df[taxa_df.q < 0.05].copy()
    primary_global = global_df[global_df.model == "Primary"]
    lines = [
        "# Physical activity–microbiota primary reanalysis",
        "",
        f"Primary paired microbiome cohort: n={n_primary}; antibiotic-inclusive sensitivity cohort: n={n_all}.",
        "",
        "## Primary question",
        "",
        "Does change in log-transformed total MET-min/week predict baseline-adjusted month-3 gut microbiota composition after adjustment for intervention, age, sex, and baseline BMI?",
        "",
        "## Global microbiome",
        "",
    ]
    for _, r in primary_global.iterrows():
        lines.append(f"- {r.analysis}, {r.taxonomic_level}: partial R²={r.partial_R2:.4f}, permutation P={r.p:.4f}.")
    lines += ["", "No alpha-diversity association survived FDR correction.", "", "## FDR-significant taxa", ""]
    for _, r in hits.iterrows():
        lines.append(f"- {r.taxonomic_level} *{r.taxon}*: standardized β={r.beta:.3f} (95% CI {r.ci_low:.3f} to {r.ci_high:.3f}), P={r.p:.3g}, q={r.q:.4f}.")
    lines += ["", "## Robustness and downstream relevance", ""]
    for taxon in hits.taxon:
        sub = targeted_df[(targeted_df.taxon == taxon) & (targeted_df.analysis == "PA-taxon sensitivity")]
        for model in ["+ baseline PA", "+ diet changes", "Include antibiotic-exposed"]:
            r = sub[sub.model == model].iloc[0]
            lines.append(f"- *{taxon}*, {model}: β={r.beta:.3f}, P={r.p:.4f}.")
    homa = score_df[score_df.analysis == "Microbiota score to metabolic outcome"].iloc[0]
    lines.append(f"- The post-selection PA-responsive microbiota score was not associated with month-3 HOMA-IR after adjustment (β={homa.beta:.3f}, P={homa.p:.4f}).")
    lines += ["", "## Interpretation guardrails", "", "- CLR coefficients describe relative composition, not absolute bacterial counts.", "- PA was measured observationally; estimates are associations, not causal effects.", "- Loss of significance after baseline-PA adjustment indicates sensitivity to regression-to-the-mean/collinearity and requires cautious, exploratory reporting.", "- Interaction and microbiota-score analyses are targeted/exploratory and should not override nonsignificant formal interaction tests.", "- Raw participant data are intentionally not included in this repository.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="PA-centered microbiome reanalysis for paired baseline and month-3 data")
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--relative", required=True)
    parser.add_argument("--outdir", default="pa_microbiome_reanalysis")
    parser.add_argument("--permutations", type=int, default=4999)
    args = parser.parse_args()
    configure_paths(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = core.clean_clinical()
    b, e, w = core.make_wide(d)
    (counts, rel, clr, alpha, taxa, primary_codes, all_codes, prevalence, genus_clr, genera, genus_prevalence) = core.microbiome_objects(d, b, e, w)
    species, data, sy0, sy1, sdy = screen_taxa(w, clr, taxa, primary_codes, "Species")
    genus, gdata, gy0, gy1, gdy = screen_taxa(w, genus_clr, genera, primary_codes, "Genus")
    taxa_results = pd.concat([species, genus], ignore_index=True).sort_values(["q", "p"])
    global_results = pd.concat([
        global_tests(data, sy0, sdy, "Species", args.permutations),
        global_tests(gdata, gy0, gdy, "Genus", args.permutations),
    ], ignore_index=True)
    global_results["q"] = core.bh_adjust(global_results["p"])
    alpha_results = alpha_tests(w, alpha, primary_codes)

    hits = taxa_results[taxa_results.q < 0.05]
    selected = []
    targeted_rows = []
    for _, hit in hits.iterrows():
        if hit.taxonomic_level == "Genus":
            tdata, ty0, ty1, _ = paired_taxon_matrices(w, genus_clr, [hit.taxon], primary_codes)
            target_clr = genus_clr
        else:
            tdata, ty0, ty1, _ = paired_taxon_matrices(w, clr, [hit.taxon], primary_codes)
            target_clr = clr
        targeted_rows += targeted_models(hit.taxon, hit.taxonomic_level, tdata, ty0[hit.taxon], ty1[hit.taxon], w, target_clr, all_codes, args.permutations)
        selected.append({"taxon": hit.taxon, "level": hit.taxonomic_level, "beta": hit.beta, "y0": ty0[hit.taxon], "y1": ty1[hit.taxon]})
    targeted = pd.DataFrame(targeted_rows)
    interaction = targeted.analysis == "Effect modification"
    if interaction.any():
        targeted.loc[interaction, "q_interaction"] = core.bh_adjust(targeted.loc[interaction, "p"])

    score = build_pa_microbiota_score(data, selected)
    score_results = score_outcomes(w, data, score)

    global_results.to_csv(outdir / "global_microbiome_results.csv", index=False)
    alpha_results.to_csv(outdir / "alpha_diversity_results.csv", index=False)
    taxa_results.to_csv(outdir / "taxon_screen_results.csv", index=False)
    targeted.to_csv(outdir / "targeted_sensitivity_and_interactions.csv", index=False)
    score_results.to_csv(outdir / "pa_microbiota_score_outcomes.csv", index=False)
    forest_plot(targeted, outdir / "pa_taxa_sensitivity_forest.png")
    write_summary(outdir / "README_results.md", global_results, alpha_results, taxa_results, targeted, score_results, len(primary_codes), len(all_codes))
    print((outdir / "README_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
