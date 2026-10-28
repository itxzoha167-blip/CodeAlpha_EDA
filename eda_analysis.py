
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "StudentsPerformance.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------
df = pd.read_csv(DATA_FILE)

required_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "math score",
    "reading score",
    "writing score",
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("=" * 60)
print("CODEALPHA TASK 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print("\nData types:")
print(df.dtypes)

# --------------------------------------------------
# 3. DATA QUALITY CHECKS
# --------------------------------------------------
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

score_columns = ["math score", "reading score", "writing score"]

print("\nScores outside the expected 0-100 range:")
for column in score_columns:
    invalid = df[~df[column].between(0, 100)]
    print(f"{column}: {len(invalid)}")

# --------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# --------------------------------------------------
summary = df[score_columns].describe().round(2)
summary.to_csv(OUTPUT_DIR / "summary_statistics.csv")

print("\nSummary statistics:")
print(summary)

print("\nAverage score by gender:")
print(df.groupby("gender")[score_columns].mean().round(2))

print("\nAverage scores by test preparation:")
print(
    df.groupby("test preparation course")[score_columns]
    .mean()
    .round(2)
)

# --------------------------------------------------
# 5. OUTLIER DETECTION USING IQR
# --------------------------------------------------
print("\nPotential outliers using the IQR method:")

outlier_results = []

for column in score_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    outlier_results.append({
        "subject": column,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": len(outliers),
    })

    print(f"{column}: {len(outliers)} potential outliers")

pd.DataFrame(outlier_results).to_csv(
    OUTPUT_DIR / "outlier_analysis.csv",
    index=False
)

# --------------------------------------------------
# 6. HYPOTHESIS TESTING
# --------------------------------------------------
# Compare average math scores between students who
# completed preparation and those who did not.
# Welch's t-test does not assume equal variances.

try:
    from scipy.stats import ttest_ind

    prepared = df.loc[
        df["test preparation course"] == "completed",
        "math score"
    ]

    not_prepared = df.loc[
        df["test preparation course"] == "none",
        "math score"
    ]

    statistic, p_value = ttest_ind(
        prepared,
        not_prepared,
        equal_var=False
    )

    print("\nHypothesis test: Math scores and test preparation")
    print(f"Prepared group mean: {prepared.mean():.2f}")
    print(f"Unprepared group mean: {not_prepared.mean():.2f}")
    print(f"Welch's t-statistic: {statistic:.3f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print(
            "Result: Evidence of a difference in average math "
            "scores between the two groups (5% significance level)."
        )
    else:
        print(
            "Result: Insufficient evidence of a difference in "
            "average math scores at the 5% significance level."
        )

    pd.DataFrame([{
        "prepared_mean": prepared.mean(),
        "not_prepared_mean": not_prepared.mean(),
        "t_statistic": statistic,
        "p_value": p_value,
        "significance_level": 0.05,
    }]).to_csv(
        OUTPUT_DIR / "hypothesis_test.csv",
        index=False
    )

except ImportError:
    print(
        "\nSciPy is not installed. Install it with "
        "'python -m pip install scipy' to run the hypothesis test."
    )

# --------------------------------------------------
# 7. VISUALIZATIONS
# --------------------------------------------------

# Chart 1: Score distributions
plt.figure(figsize=(9, 5))

for column in score_columns:
    sns.histplot(
        data=df,
        x=column,
        label=column.title(),
        kde=True,
        alpha=0.3
    )

plt.title("Distribution of Student Exam Scores")
plt.xlabel("Score")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "score_distributions.png", dpi=300)
plt.close()

# Chart 2: Average score by test preparation
preparation_means = df.groupby(
    "test preparation course"
)[score_columns].mean()

preparation_means.T.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Average Exam Scores by Test Preparation")
plt.xlabel("Subject")
plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.legend(title="Test Preparation")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "preparation_comparison.png", dpi=300)
plt.close()

# Chart 3: Correlation heatmap
plt.figure(figsize=(7, 5))

correlation = df[score_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f"
)

plt.title("Correlation Between Exam Scores")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "score_correlation.png", dpi=300)
plt.close()

# Chart 4: Score boxplots
plt.figure(figsize=(8, 5))

sns.boxplot(data=df[score_columns])

plt.title("Exam Score Spread and Potential Outliers")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "score_boxplots.png", dpi=300)
plt.close()

# --------------------------------------------------
# 8. EXPORT CLEANED DATA AND FINAL SUMMARY
# --------------------------------------------------
df.to_csv(OUTPUT_DIR / "analyzed_student_data.csv", index=False)

report = f"""# Exploratory Data Analysis Summary

- Dataset rows: {df.shape[0]}
- Dataset columns: {df.shape[1]}
- Duplicate rows: {df.duplicated().sum()}
- Total missing cells: {df.isnull().sum().sum()}

## Average Exam Scores

{df[score_columns].mean().round(2).to_string()}

## Key Notes

- The dataset includes math, reading, and writing scores.
- Group comparisons describe associations, not proof of causation.
- IQR flags potential outliers, which should be reviewed rather
  than automatically removed.
- Statistical test results are saved separately when SciPy is installed.
"""

(OUTPUT_DIR / "eda_summary.md").write_text(
    report,
    encoding="utf-8"
)

print("\nAnalysis completed successfully.")
print(f"Results and charts saved in: {OUTPUT_DIR}")
