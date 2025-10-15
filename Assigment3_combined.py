# ============================================================
# HIT140 Project – S225 FOUNDATIONS OF DATA SCIENCE. Assignment 3
#Bidhan Chaudhary (S394807)
#Dipesh Sedhai (S395457)
#Nitesh Raj Chaudhary (S394811)
#Roshan Kumar Shrestha (S395498)
# Datasets: dataset1.csv (bat landings, per-event) & dataset2.csv (30-min windows)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor # Added for VIF

# sklearn usage (used for simple train/test R²)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    print("Note: scikit-learn not available. Sklearn-based evaluation will be skipped. "
          "Install via `pip install scikit-learn` if you want train/test evaluation.")

# ---------- Settings ----------
pd.set_option('display.max_columns', 200)
sns.set(style="whitegrid", context="talk")


# ---------- Utility: save figure ----------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------- Utility: Season mapping helper ----------
def season_from_month(m):
    """Maps month number to a season name (Northern Hemisphere)."""
    if pd.isna(m): return np.nan
    try: mm = int(m)
    except Exception: return np.nan
    if mm in [12, 1, 2]: return "Winter"
    if mm in [3, 4, 5]: return "Spring"
    if mm in [6, 7, 8]: return "Summer"
    if mm in [9, 10, 11]: return "Autumn"
    return np.nan


# ----------  Load datasets ----------
df1 = pd.read_csv("dataset1.csv")   # event-level
df2 = pd.read_csv("dataset2.csv")   # 30-min window-level


# ----------  Cleaning functions ----------
def clean_dataset1(df1):
    """Clean event-level dataset: parse times, coerce numerics, drop missing critical rows,
       winsorize numeric extremes, clean habit labels, create features."""
    df = df1.copy()

    # 1. Parse dates (if present)
    for col in ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # *** FIX: Robustly derive month and season from start_time ***
    if "start_time" in df.columns:
        # Extract month number from parsed datetime
        df["month_num"] = df["start_time"].dt.month
        # Map season using the reliable month number
        df["season"] = df["month_num"].apply(season_from_month)
    else:
        # If start_time is missing, fall back to original 'month' column if present
        if "month" in df.columns:
            df["month_num"] = pd.to_numeric(df["month"], errors="coerce")
            df["season"] = df["month_num"].apply(season_from_month)
        else:
             df["season"] = np.nan
    # ***************************************************************

    #  Coerce numeric columns
    numeric_cols = [
        "bat_landing_to_food", "seconds_after_rat_arrival",
        "risk", "reward", "month", "hours_after_sunset" # 'season' removed as it's now derived
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows missing critical fields for event analysis
    required = ["start_time", "bat_landing_to_food", "risk", "reward"]
    present_required = [c for c in required if c in df.columns]
    if present_required:
        df = df.dropna(subset=present_required)

    # Remove negative/implausible values and winsorize extremes
    if "bat_landing_to_food" in df.columns:
        df = df[df["bat_landing_to_food"] >= 0]
        q1, q99 = df["bat_landing_to_food"].quantile([0.01, 0.99])
        df["bat_landing_to_food"] = df["bat_landing_to_food"].clip(lower=q1, upper=q99)

    if "seconds_after_rat_arrival" in df.columns:
        q1, q99 = df["seconds_after_rat_arrival"].quantile([0.01, 0.99])
        df["seconds_after_rat_arrival"] = df["seconds_after_rat_arrival"].clip(lower=q1, upper=q99)

    # Habit cleaning / normalization (no change)
    if "habit" in df.columns:
        mask_clean = ~df["habit"].astype(str).str.contains(r",", regex=True, na=False)
        df.loc[mask_clean, "habit_cleaned"] = df.loc[mask_clean, "habit"].astype(str)

        replace_map = {
            "bat_figiht": "bat_fight", "rat attack": "rat_attack", "bats": "bat",
            "others": "other", "other_bats": "other_bat", "other_bats/rat": "other_bat_rat",
            "other directions": "other", "fight_bat": "bat_fight", "rat_and_bat": "bat_and_rat",
            "pick_and_rat": "rat_and_pick", "pick_and_bat": "bat_and_pick",
            "rat_pick_and_bat": "pick_rat_and_bat", "bat_rat_pick": "pick_rat_bat",
            "pick_bat_rat": "pick_rat_bat", "eating_bat_rat_pick": "eating_bat_pick",
            "bat_fight_and_rat": "bat_fight_rat", "rat_bat_fight": "bat_fight_rat",
            "pick_and_all": "all_pick", " ": "unknown"
        }
        df["habit_cleaned"] = df["habit_cleaned"].astype(str).str.lower().str.replace(r"[ ,/]", "_", regex=True)
        df["habit_cleaned"] = df["habit_cleaned"].replace(replace_map)
        df["habit_cleaned"] = df["habit_cleaned"].fillna("unknown")
    else:
        df["habit_cleaned"] = "unknown"

    # Flag if rat present at landing time (no change)
    if "seconds_after_rat_arrival" in df.columns:
        df["rat_present_at_landing"] = (df["seconds_after_rat_arrival"] >= 0).astype(int)

    # Create vigilance proxy (no change)
    if "bat_landing_to_food" in df.columns:
        df["vigilance_delay_s"] = df["bat_landing_to_food"]
    else:
        df["vigilance_delay_s"] = np.nan

    return df


def clean_dataset2(df2):
    """Clean window-level dataset: parse times, coerce numerics, drop missing critical rows,
       filter rat_minutes plausible range, winsorize extremes, create rat_present."""
    df = df2.copy()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", dayfirst=True)
        # *** FIX: Robustly derive month and season from time ***
        df["month_num"] = df["time"].dt.month
        df["season"] = df["month_num"].apply(season_from_month)
        # ********************************************************
    else:
        # Fallback if time is missing
        if "month" in df.columns:
            df["month_num"] = pd.to_numeric(df["month"], errors="coerce")
            df["season"] = df["month_num"].apply(season_from_month)
        else:
             df["season"] = np.nan

    numeric_cols = [
        "hours_after_sunset", "bat_landing_number",
        "food_availability", "rat_minutes", "rat_arrival_number", "month"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()

    # Require core fields for window-level analyses (no change)
    required = ["time", "bat_landing_number", "rat_minutes"]
    present_req = [c for c in required if c in df.columns]
    if present_req:
        df = df.dropna(subset=present_req)

    # Keep rat_minutes within a plausible window (0-30 minutes) (no change)
    if "rat_minutes" in df.columns:
        df = df[df["rat_minutes"].between(0, 30)]

    # Remove negatives and winsorize extremes on key numeric columns (no change)
    for col in ["rat_arrival_number", "food_availability", "bat_landing_number"]:
        if col in df.columns:
            df = df[df[col] >= 0]
            q1, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=q1, upper=q99)

    # features (no change)
    if "rat_minutes" in df.columns:
        df["rat_present"] = (df["rat_minutes"] > 0).astype(int)
        df["rat_pressure"] = df["rat_minutes"] / 30.0

    return df


df1_clean = clean_dataset1(df1)
df2_clean = clean_dataset2(df2)




#################### Feature engineering ####################
# time period bins (window-level)
if "hours_after_sunset" in df2_clean.columns:
    df2_clean["time_period"] = pd.cut(df2_clean["hours_after_sunset"], bins=[-1, 2, 5, 10], labels=["Early", "Mid", "Late"])

# date columns for merging
if "start_time" in df1_clean.columns:
    df1_clean["date"] = pd.to_datetime(df1_clean["start_time"]).dt.date
if "time" in df2_clean.columns:
    df2_clean["date"] = pd.to_datetime(df2_clean["time"]).dt.date


###################### Merge by DATE (daily aggregate of window-level df2) #################
def merge_by_date(d1, d2):
    """
    Aggregate window-level d2 to daily summaries and left-merge onto event-level d1 by date.
    """
    d1 = d1.copy()
    d2 = d2.copy()
    d1["date"] = pd.to_datetime(d1["start_time"]).dt.date
    d2["date"] = pd.to_datetime(d2["time"]).dt.date

    # Aggregate d2 per date (choose aggregation functions)
    agg_dict = {}
    for c in ["bat_landing_number", "food_availability", "rat_minutes", "rat_arrival_number", "hours_after_sunset", "rat_pressure"]:
        if c in d2.columns:
            agg_dict[c] = "mean"
    if "rat_present" in d2.columns:
        agg_dict["rat_present"] = "max"  # 1 if rats present any time that day
    if "month_num" in d2.columns: # Use the reliable month_num for 'first'
        agg_dict["month_num"] = "first"
    if "season" in d2.columns:
        agg_dict["season"] = "first"

    agg_d2 = d2.groupby("date").agg(agg_dict).reset_index()

    merged = pd.merge(d1, agg_d2, on="date", how="left", suffixes=("_event", "_day"))
    return merged


# Use date-level merge by default
df_merged = merge_by_date(df1_clean, df2_clean)


print("Merged (date-level) shape:", df_merged.shape)


#  Exploratory Visuals & Tests ----------
#  Average landing delay by habit
if "habit_cleaned" in df1_clean.columns and "vigilance_delay_s" in df1_clean.columns:
    avg_landing = df1_clean.groupby("habit_cleaned")["vigilance_delay_s"].mean().sort_values()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=avg_landing.index, y=avg_landing.values, palette="viridis")
    plt.title("Average Landing Delay (Seconds) by Bat Behaviour Type", fontsize=14)
    plt.xlabel("Bat Behaviour (Cleaned Habit Categories)", fontsize=12)
    plt.ylabel("Average Delay Before Feeding (Seconds)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    savefig("avg_landing_time_by_habit_barchart.png")

# Risk & Reward stacked by habit
if "habit_cleaned" in df1_clean.columns and "risk" in df1_clean.columns and "reward" in df1_clean.columns:
    ct = pd.crosstab(df1_clean["habit_cleaned"], [df1_clean["risk"], df1_clean["reward"]])
    try:
        ct = ct.reorder_levels([0, 1], axis=1).sort_index(axis=1)
    except Exception:
        pass
    labels = []
    for col in ct.columns:
        r, rw = col
        labels.append(f"{'High' if r == 1 else 'Low'} Risk / {'High' if rw == 1 else 'Low'} Reward")
    ct.columns = labels
    plt.figure(figsize=(13, 7))
    ct.plot(kind="bar", stacked=True, colormap="coolwarm", figsize=(13, 7), ax=plt.gca()) # Added ax=plt.gca()
    plt.title("Distribution of Risk and Reward by Bat Behaviour Category", fontsize=14)
    plt.xlabel("Bat Behaviour (Cleaned Habit Categories)", fontsize=12)
    plt.ylabel("Number of Observations", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.legend(
        title="Risk and Reward Combination",
        labels=["Low Risk / Low Reward", "Low Risk / High Reward",
                "High Risk / Low Reward", "High Risk / High Reward"],
        bbox_to_anchor=(1.02, 1), loc="upper left"
    )
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    savefig("risk_reward_by_habit_stackedbarchart.png")

# Correlation and scatter for 'rat' habit
if "habit_cleaned" in df1_clean.columns:
    df1_rat = df1_clean[df1_clean["habit_cleaned"].astype(str).str.lower().eq("rat")]
    if not df1_rat.empty and "seconds_after_rat_arrival" in df1_rat.columns and "vigilance_delay_s" in df1_rat.columns:
        sub = df1_rat[["seconds_after_rat_arrival", "vigilance_delay_s"]].dropna()
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub["seconds_after_rat_arrival"], sub["vigilance_delay_s"])
            print(f"[dataset1] Pearson corr (seconds_after_rat_arrival vs landing delay): r={r:.3f}, p={p:.4f}")
            plt.figure(figsize=(9, 6))
            plt.scatter(sub["seconds_after_rat_arrival"], sub["vigilance_delay_s"], alpha=0.6)
            plt.title('Landing Delay vs Seconds After Rat Arrival (habit="rat")')
            plt.xlabel("Seconds after rat arrival")
            plt.ylabel("Landing delay (s)")
            savefig("landing_delay_vs_seconds_after_rat_arrival.png")

#  Boxplots on window-level by rat presence
if "rat_present" in df2_clean.columns:
    if "bat_landing_number" in df2_clean.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="rat_present", y="bat_landing_number", data=df2_clean)
        plt.title("Bat Landings per 30-min Window vs Rat Presence")
        plt.xlabel("Rat present (0/1)")
        plt.ylabel("Bat landings")
        savefig("box_bat_landings_vs_rat_presence.png")

    if "food_availability" in df2_clean.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="rat_present", y="food_availability", data=df2_clean)
        plt.title("Food Availability vs Rat Presence")
        plt.xlabel("Rat present (0/1)")
        plt.ylabel("Food availability")
        savefig("box_food_vs_rat_presence.png")

# T-tests (window-level): bat_landing_number and food_availability
if "rat_present" in df2_clean.columns:
    with_rats = df2_clean[df2_clean["rat_present"] == 1]
    without_rats = df2_clean[df2_clean["rat_present"] == 0]
    if not with_rats.empty and not without_rats.empty:
        if "bat_landing_number" in df2_clean.columns:
            t1, p1 = stats.ttest_ind(with_rats["bat_landing_number"], without_rats["bat_landing_number"], equal_var=False, nan_policy="omit")
            print(f"\n[dataset2] T-test bat_landing_number: t={t1:.3f}, p={p1:.4f}")
        if "food_availability" in df2_clean.columns:
            t2, p2 = stats.ttest_ind(with_rats["food_availability"], without_rats["food_availability"], equal_var=False, nan_policy="omit")
            print(f"[dataset2] T-test food_availability: t={t2:.3f}, p={p2:.4f}")


# ---------- Summary tables ----------
if "habit_cleaned" in df1_clean.columns:
    summary_habit = df1_clean.groupby("habit_cleaned")[["risk", "reward", "vigilance_delay_s"]].mean().sort_values("risk", ascending=False)
    print("\n[dataset1] Mean risk, reward, delay by habit (top 10):\n", summary_habit.head(10))

if "rat_present" in df2_clean.columns:
    summary_window = df2_clean.groupby("rat_present")[["bat_landing_number", "food_availability", "rat_minutes"]].mean()
    print("\n[dataset2] Means by rat presence:\n", summary_window)

# ----------  Investigation A: Linear Regression (OLS) ----------
response = "vigilance_delay_s"
# Candidate predictors (using the non-suffixed names which are correct post-merge)
candidate_predictors = [
    "rat_minutes", "rat_pressure", "rat_arrival_number",
    "food_availability", "hours_after_sunset", "bat_landing_number"
]
# select predictors actually present in merged frame
predictors = [p for p in candidate_predictors if p in df_merged.columns]

# Prepare dataset for LR
dataA = df_merged[[response] + predictors].dropna()
print(f"\n[Investigation A] rows available for LR: {len(dataA)}   Predictors used: {predictors}")

if len(dataA) >= 5 and predictors:
    X = dataA[predictors]
    y = dataA[response]

    # Optional sklearn evaluation
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        print("\n[Investigation A] Sklearn LR evaluation (test):")
        print(" R2 (test):", r2_score(y_test, y_pred))
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(" RMSE (test):", rmse)

    # Statsmodels OLS (full-sample) for inference
    X_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const).fit()
    print("\n[Investigation A] OLS summary:\n")
    print(ols_model.summary())

    # Residual diagnostics: histogram and residuals vs fitted
    plt.figure(num=1, figsize=(8,6))
    sns.histplot(ols_model.resid, kde=True, color='skyblue')
    plt.title("Investigation A: Residuals Distribution", fontsize=14)
    plt.xlabel("Residuals (Observed - Predicted Vigilance Delay, s)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    savefig("investA_residuals_hist.png")

    plt.figure(num=2, figsize=(8,6))
    plt.scatter(ols_model.fittedvalues, ols_model.resid, alpha=0.6, color='orange')
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Investigation A: Residuals vs Fitted Values", fontsize=14)
    plt.xlabel("Fitted values (Predicted Vigilance Delay, s)", fontsize=12)
    plt.ylabel("Residuals (Observed - Predicted, s)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    savefig("investA_resid_vs_fitted.png")

    # Multicollinearity: correlation heatmap and VIF
    plt.figure(figsize=(8,6))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Predictor correlation (Investigation A)")
    savefig("investA_corr_heatmap.png")

    X_vif = sm.add_constant(X)
    vif_df = pd.DataFrame({
        "variable": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    })
    print("\n[Investigation A] VIFs:\n", vif_df.to_string(index=False))
else:
    print("\n[Investigation A] Not enough data or no predictors available for LR.")


# Add predicted values for plotting (if OLS model exists)
if 'ols_model' in globals():
    df_plot = df_merged.copy()
    X_const_full = sm.add_constant(df_merged[predictors].fillna(X.mean())) # Use mean imputation for plotting predictions on full data
    # Predict on the full (or imputed) dataset for visualization
    df_plot["predicted_vigilance"] = ols_model.predict(X_const_full.dropna())


# ---------- Investigation B: Seasonal comparison (Winter vs Spring) ----------
# Ensure we have a consistent season column in merged data
if "season" in df_merged.columns:
    df_merged["season_use"] = df_merged["season"]
elif "season_event" in df_merged.columns:
    df_merged["season_use"] = df_merged["season_event"]
else:
    # fallback to mapping from start_time month
    if "start_time" in df_merged.columns:
        df_merged["season_use"] = pd.to_datetime(df_merged["start_time"]).dt.month.apply(season_from_month)
    else:
        df_merged["season_use"] = np.nan

# Filter to Winter and Spring only
df_b = df_merged[df_merged["season_use"].isin(["Winter", "Spring"])].copy()
print("\n[Investigation B] counts by season (Winter/Spring):\n", df_b["season_use"].value_counts().to_dict())

# prepare data for seasonal OLS (dropna for required cols)
if predictors:
    df_b_full = df_b[[response] + predictors + ["season_use"]].dropna()
else:
    df_b_full = pd.DataFrame()

def fit_by_season(df, season_label):
    df_s = df[df["season_use"] == season_label]
    if len(df_s) < 10:
        print(f"WARNING: small sample for {season_label}: n={len(df_s)}")
        return None
    Xs = df_s[predictors]
    ys = df_s[response]
    Xs_const = sm.add_constant(Xs)
    model = sm.OLS(ys, Xs_const).fit()
    return model

if not df_b_full.empty:
    model_winter = fit_by_season(df_b_full, "Winter")
    model_spring = fit_by_season(df_b_full, "Spring")

    if model_winter and model_spring:
        print("\n[Investigation B] Winter model summary:\n", model_winter.summary())
        print("\n[Investigation B] Spring model summary:\n", model_spring.summary())

        # Compare coefficients
        coefs = pd.DataFrame({
            "var": model_winter.params.index,
            "coef_winter": model_winter.params.values,
            "se_winter": model_winter.bse.values,
        }).merge(pd.DataFrame({
            "var": model_spring.params.index,
            "coef_spring": model_spring.params.values,
            "se_spring": model_spring.bse.values,
        }), on="var", how="outer").fillna(0)

        coefs["coef_diff"] = coefs["coef_winter"] - coefs["coef_spring"]
        coefs["se_diff"] = np.sqrt(coefs["se_winter"]**2 + coefs["se_spring"]**2)
        coefs["z_diff"] = coefs["coef_diff"] / coefs["se_diff"].replace(0, np.nan)
        coefs["p_diff_two_sided"] = 2 * (1 - stats.norm.cdf(np.abs(coefs["z_diff"])))
        print("\n[Investigation B] Coefficient difference (Winter - Spring):\n", coefs[["var","coef_winter","coef_spring","coef_diff","se_diff","z_diff","p_diff_two_sided"]].to_string(index=False))

        # Plot coefficient comparison
        plot_df = coefs.set_index("var")[["coef_winter","coef_spring"]].drop(index='const', errors='ignore') # Drop constant for cleaner plot
        plt.figure(figsize=(10,6))
        plot_df.plot(kind="bar", ax=plt.gca(), color=["#1f77b4", "#ff7f0e"])
        plt.title("Linear Regression Coefficient Comparison: Winter vs Spring", fontsize=14)
        plt.ylabel("Coefficient Value", fontsize=12)
        plt.xlabel("Predictor Variables", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.legend(["Winter Model", "Spring Model"], title="Season", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        savefig("coeff_compare_winter_spring.png")

        # Interaction model (season * rat_minutes) - test moderator effect
        try:
            df_int = df_b_full.dropna(subset=[response, "rat_minutes", "hours_after_sunset"])
            df_int["season_cat"] = df_int["season_use"].astype("category")
            # Only include predictors used in the main model (except rat_minutes which is used for interaction)
            covariates = [p for p in predictors if p not in ["rat_minutes", "rat_pressure"]]
            formula = f"{response} ~ rat_minutes * season_cat + {' + '.join(covariates)}"
            model_inter = smf.ols(formula, data=df_int).fit()
            print("\n[Investigation B] Interaction model summary:\n", model_inter.summary())
            interaction_terms = [term for term in model_inter.pvalues.index if ":" in term]
            if interaction_terms:
                print("\n[Investigation B] Interaction term p-values:\n", model_inter.pvalues.loc[interaction_terms])
            else:
                print("\n[Investigation B] No interaction terms found in model formula.\n")
        except Exception as e:
            print(" Interaction model failed due to:", e)
    else:
        print("\n[Investigation B] Models could not be fitted due to small sample size in one or both seasons.")
else:
    print("\n[Investigation B] Not enough seasonal data to fit models.")


# ----------  Final reporting summary ----------
print("\n--- Final reporting: top-level summaries ---")
print("Dataset1 (event-level) rows (clean):", len(df1_clean))
print("Dataset2 (window-level) rows (clean):", len(df2_clean))
print("Merged (date-level) rows:", len(df_merged))

if 'ols_model' in globals():
    print("\n[Investigation A] OLS coefficients (full sample):")
    print(ols_model.params.to_string())

if 'coefs' in globals():
    print("\n[Investigation B] Coefficient differences (Winter - Spring):")
    print(coefs[["var","coef_diff","p_diff_two_sided"]].to_string(index=False))

print("\nScript complete. Plots saved as PNGs for slides.")



# -------------------------------
# Observed vs Predicted Visualizations
# -------------------------------

df_plot = df_merged.copy()

if 'ols_model' in globals():
    df_plot["predicted_vigilance"] = ols_model.fittedvalues
else:
    raise ValueError("OLS model not found. Run Investigation A first.")

# Observed vs Predicted over Date
if "date" in df_plot.columns:
    plt.figure(figsize=(12,6))
    plt.plot(df_plot["date"], df_plot["vigilance_delay_s"], label="Observed", marker='o', linestyle='None', alpha=0.6)
    plt.plot(df_plot["date"], df_plot["predicted_vigilance"], label="Predicted", marker='x', linestyle='-', color='red')
    plt.xlabel("Date")
    plt.ylabel("Vigilance Delay (s)")
    plt.title("Observed vs Predicted Vigilance Delay Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    savefig("obs_vs_pred_date.png")


# Observed vs Predicted by Season
if "season_use" in df_plot.columns:
    plt.figure(figsize=(8,6))
    obs_means = df_plot.groupby("season_use")["vigilance_delay_s"].mean()
    pred_means = df_plot.groupby("season_use")["predicted_vigilance"].mean()
    df_season_compare = pd.DataFrame({"Observed": obs_means, "Predicted": pred_means}).reset_index()
    
    x = np.arange(len(df_season_compare))
    width = 0.35
    plt.bar(x - width/2, df_season_compare["Observed"], width, label="Observed", color="skyblue")
    plt.bar(x + width/2, df_season_compare["Predicted"], width, label="Predicted", color="salmon")
    plt.xticks(x, df_season_compare["season_use"])
    plt.ylabel("Vigilance Delay (s)")
    plt.title("Observed vs Predicted Vigilance Delay by Season")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    savefig("obs_vs_pred_season.png")



print(" Observed vs Predicted plots generated and saved.")
