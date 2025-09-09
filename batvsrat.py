# ============================================================
# HIT140 Project – S225 FOUNDATIONS OF DATA SCIENCE
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

# ---------- Settings ----------
pd.set_option('display.max_columns', 200)
sns.set(style="whitegrid", context="talk")

# ---------- 0) Utility: Save figure helper ----------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()

# ---------- 1) Load ----------
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

# print(df1.head)
# print(df1.describe)


# ---------- 2) Cleaning: dataset1 ----------
def clean_dataset1(df1):
    df = df1.copy()

    # a) Parse times (coerce errors -> NaT)
    #    If your CSV stores these in a different column, adjust names accordingly.
    for col in ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # b) Enforce numeric types where expected (ignore missing columns gracefully)
    numeric_cols = [
        "bat_landing_to_food", "seconds_after_rat_arrival",
        "risk", "reward", "month", "hours_after_sunset", "season"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # c) Drop full duplicates
    df = df.drop_duplicates()

    # d) Handle missing critical fields
    #    For event-level analysis we need at least: start_time, bat_landing_to_food, risk, reward
    df = df.dropna(subset=["start_time", "bat_landing_to_food", "risk", "reward"])

    # e) Outlier handling (domain rules; tune as needed)
    #    Remove negative/implausible values
    if "bat_landing_to_food" in df.columns:
        df = df[df["bat_landing_to_food"] >= 0]
        # trim extreme right tail (5th–95th / IQR — choose one). Here winsorize to 1st–99th pct
        q1, q99 = df["bat_landing_to_food"].quantile([0.01, 0.99])
        df["bat_landing_to_food"] = df["bat_landing_to_food"].clip(lower=q1, upper=q99)

    if "seconds_after_rat_arrival" in df.columns:
        # seconds after rat arrival can be negative if bat arrived before rat; keep within sensible bounds
        q1, q99 = df["seconds_after_rat_arrival"].quantile([0.01, 0.99])
        df["seconds_after_rat_arrival"] = df["seconds_after_rat_arrival"].clip(lower=q1, upper=q99)

    # f) Habit cleaning (typos & standardization)
    if "habit" in df.columns:
        # remove entries with commas (mixed/ambiguous codes)
        mask_clean = ~df["habit"].astype(str).str.contains(r",", regex=True, na=False)
        df.loc[mask_clean, "habit_cleaned"] = df.loc[mask_clean, "habit"].astype(str)

        replace_map = {
            "bat_figiht": "bat_fight",
            "rat attack": "rat_attack",
            "bats": "bat",
            "others": "other",
            "other_bats": "other_bat",
            "other_bats/rat": "other_bat_rat",
            "other directions": "other",
            "fight_bat": "bat_fight",
            "rat_and_bat": "bat_and_rat",
            "pick_and_rat": "rat_and_pick",
            "pick_and_bat": "bat_and_pick",
            "rat_pick_and_bat": "pick_rat_and_bat",
            "bat_rat_pick": "pick_rat_bat",
            "pick_bat_rat": "pick_rat_bat",
            "eating_bat_rat_pick": "eating_bat_pick",
            "bat_fight_and_rat": "bat_fight_rat",
            "rat_bat_fight": "bat_fight_rat",
            "pick_and_all": "all_pick",
            " ": "unknown"
        }
        df["habit_cleaned"] = df["habit_cleaned"].replace(replace_map)
        df["habit_cleaned"] = df["habit_cleaned"].fillna("unknown")
    else:
        df["habit_cleaned"] = "unknown"

    # g) Binary flags (features)
    if "seconds_after_rat_arrival" in df.columns:
        df["rat_present_at_landing"] = (df["seconds_after_rat_arrival"] >= 0).astype(int)

    # h) Derive vigilance proxy (alias for readability)
    df["vigilance_delay_s"] = df["bat_landing_to_food"]

    return df

df1_clean = clean_dataset1(df1)

# ---------- 3) Cleaning: dataset2 ----------
def clean_dataset2(df2):
    df = df2.copy()

    # a) Parse window time
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", dayfirst=True)

    # b) Numeric coercion
    numeric_cols = [
        "hours_after_sunset", "bat_landing_number",
        "food_availability", "rat_minutes", "rat_arrival_number", "month"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # c) Drop duplicates
    df = df.drop_duplicates()

    # d) Missing critical fields (need time, bat_landing_number, rat_minutes for core analysis)
    df = df.dropna(subset=["time", "bat_landing_number", "rat_minutes"])

    # # e) Range/outlier rules (tune to your data collection protocol)
    # if "hours_after_sunset" in df.columns:
    #     df = df[df["hours_after_sunset"] >= 0]

    if "rat_minutes" in df.columns:
        # observation windows are typically 30 minutes
        df = df[df["rat_minutes"].between(0, 30)]

    for col in ["rat_arrival_number", "food_availability", "bat_landing_number"]:
        if col in df.columns:
            df = df[df[col] >= 0]
            # smothened extremes to reduce undue influence of rare spikes
            q1, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=q1, upper=q99)

    # f) Features
    df["rat_present"] = (df["rat_minutes"] > 0).astype(int)


    return df

df2_clean = clean_dataset2(df2)

# ---------- 4) Feature Engineering (extra, cross-dataset-agnostic) ----------
# Normalize/scale-like features (kept simple here)
if "rat_minutes" in df2_clean.columns:
    df2_clean["rat_pressure"] = df2_clean["rat_minutes"] / 30.0  # fraction of window

# ---------- 5) ANALYSIS: dataset1 (event-level) ----------

# A) Habit vs average landing delay (vigilance)
avg_landing = df1_clean.groupby("habit_cleaned")["vigilance_delay_s"].mean().sort_values()
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_landing.index, y=avg_landing.values, palette="viridis")
plt.title("Average Landing Delay (s) by Habit")
plt.xlabel("Habit (cleaned)")
plt.ylabel("Average delay (s)")
plt.xticks(rotation=45, ha="right")
savefig("avg_landing_time_by_habit_barchart.png")

# B) Risk & Reward distribution by habit (stacked)
ct = pd.crosstab(df1_clean["habit_cleaned"], [df1_clean["risk"], df1_clean["reward"]])

# Re-order & flatten column labels to intuitive order:
# (0,0)=Low risk/Low reward,(0,1)=Low risk/High reward, (1,0)=High risk/Low reward,  (1,1)=High risk/High reward
ct = ct.reorder_levels([0, 1], axis=1).sort_index(axis=1)
ct.columns = [
    "Low Risk / Low Reward",
    "Low Risk / High Reward",
    "High Risk / Low Reward",
    "High Risk / High Reward",
]
plt.figure(figsize=(13, 7))
ct.plot(kind="bar", stacked=True, colormap="coolwarm", figsize=(13, 7))
plt.title("Distribution of Risk & Reward by Habit")
plt.xlabel("Habit (cleaned)")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Risk & Reward", bbox_to_anchor=(1.02, 1), loc="upper left")
savefig("risk_reward_by_habit_stackedbarchart.png")
plt.show()

# C) Correlation: rat-arrival timing vs landing delay (only records labeled as 'rat' habit if present)
if "habit" in df1_clean.columns:
    df1_rat = df1_clean[df1_clean["habit"].astype(str).str.lower().eq("rat")]
else:
    df1_rat = df1_clean[df1_clean["habit_cleaned"].str.contains("rat", case=False, na=False)]

if not df1_rat.empty and df1_rat["seconds_after_rat_arrival"].notna().any():
    # Drop rows with NaN in either column for Pearson
    sub = df1_rat[["seconds_after_rat_arrival", "vigilance_delay_s"]].dropna()
    if len(sub) >= 3:
        r, p = stats.pearsonr(sub["seconds_after_rat_arrival"], sub["vigilance_delay_s"])
        print(f"[dataset1] Pearson corr (seconds_after_rat_arrival vs landing delay): r={r:.3f}, p={p:.4f}")
    else:
        print("[dataset1] Not enough rat-habit rows for correlation.")
    # Scatter
    plt.figure(figsize=(9, 6))
    plt.scatter(sub["seconds_after_rat_arrival"], sub["vigilance_delay_s"], alpha=0.6)
    plt.title('Landing Delay vs Seconds After Rat Arrival (habit="rat")')
    plt.xlabel("Seconds after rat arrival")
    plt.ylabel("Landing delay (s)")
    savefig("landing_delay_vs_seconds_after_rat_arrival.png")

# ---------- 6) ANALYSIS: dataset2 (window-level) ----------

# A) Box plots: bat landings vs rat presence
plt.figure(figsize=(8, 6))
sns.boxplot(x="rat_present", y="bat_landing_number", data=df2_clean)
plt.title("Bat Landings per 30-min Window vs Rat Presence")
plt.xlabel("Rat present (0/1)")
plt.ylabel("Bat landings")
savefig("box_bat_landings_vs_rat_presence.png")

# B) Box plots: food availability vs rat presence
plt.figure(figsize=(8, 6))
sns.boxplot(x="rat_present", y="food_availability", data=df2_clean)
plt.title("Food Availability vs Rat Presence")
plt.xlabel("Rat present (0/1)")
plt.ylabel("Food availability")
savefig("box_food_vs_rat_presence.png")

# C) T-tests: windows with rats vs without
with_rats = df2_clean[df2_clean["rat_present"] == 1]
without_rats = df2_clean[df2_clean["rat_present"] == 0]

if not with_rats.empty and not without_rats.empty:
    t1, p1 = stats.ttest_ind(with_rats["bat_landing_number"], without_rats["bat_landing_number"], equal_var=False, nan_policy="omit")
    t2, p2 = stats.ttest_ind(with_rats["food_availability"], without_rats["food_availability"], equal_var=False, nan_policy="omit")
    print(f"[dataset2] T-test bat_landing_number: t={t1:.3f}, p={p1:.4f}")
    print(f"[dataset2] T-test food_availability: t={t2:.3f}, p={p2:.4f}")



# ----------  Nice summary tables for slides ----------

# Event-level: mean risk/reward by habit
summary_habit = df1_clean.groupby("habit_cleaned")[["risk", "reward", "vigilance_delay_s"]].mean().sort_values("risk", ascending=False)
print("\n[dataset1] Mean risk, reward, delay by habit:\n", summary_habit.head(10))

# Window-level: means by rat presence
summary_window = df2_clean.groupby("rat_present")[["bat_landing_number", "food_availability", "rat_minutes"]].mean()
print("\n[dataset2] Means by rat presence:\n", summary_window)


