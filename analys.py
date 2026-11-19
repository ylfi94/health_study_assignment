# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
df = pd.read_csv("health_study_dataset.csv")

df.columns = df.columns.str.strip()

# %%
print(df.head())
df.info()

# %%
df["sex"] = df["sex"].str.strip()
print(df["sex"].unique())

# %%
plt.hist(df["systolic_bp"], bins=15, color='steelblue', edgecolor='black')
plt.title("Fördelning av systoliskt blodtryck")
plt.xlabel("Blodtryck (mmHg)")
plt.ylabel("Frekvens")
plt.show()

# %%
df[df["sex"] == "M"]
df[df["sex"] == "F"]

weight_m = df[df["sex"] == "M"]["weight"]
weight_f = df[df["sex"] == "F"]["weight"]

print(len(weight_m))
print(len(weight_f))

print(weight_m.head())
print(weight_f.head())

plt.boxplot([weight_m, weight_f], tick_labels=["M", "F"])
plt.title("Viktfördelning efter kön")
plt.xlabel("Kön")
plt.ylabel("Vikt (kg)")
plt.show()

print(weight_m.mean())
print(weight_f.mean())

# %%
print(weight_m.median())
print(weight_f.median())

# %%
plt.scatter(df["age"], df["systolic_bp"], alpha=0.5)

plt.title("Samband mellan ålder och systoliskt blodtryck")
plt.xlabel("Ålder (år)")
plt.ylabel("Systoliskt blodtryck (mmHg)")
plt.show()

# %%
correlation = df["age"].corr(df["systolic_bp"])

print(correlation)

# %%

rng = np.random.default_rng(seed=42)

# %%
simulated_disease = rng.binomial(
    n=1,
    p=0.05875,
    size=1000
)

print(simulated_disease.sum())
# %%
simulated_proportion = simulated_disease.mean()
print(simulated_proportion)
# %%
bp_mean = df["systolic_bp"].mean()
# %%
print(bp_mean)
# %%
bp_std = df["systolic_bp"].std()
print(bp_std)
# %%
bp_count = df["systolic_bp"].count()
print(bp_count)
# %%
import math
bp_se = bp_std / math.sqrt(bp_count)
print(bp_se)

# %%
print(bp_mean)
print(bp_std)
print(bp_se)

# %%
margin_error = 1.96 * bp_se
print(margin_error)

# %%
ci_lower = bp_mean - margin_error
ci_upper = bp_mean + margin_error

print(ci_lower)
print(ci_upper)
# %%
bp_values = df["systolic_bp"].dropna().to_numpy()

print(type(bp_values))
print(len(bp_values))
# %%
bootstrap_sample = np.random.choice(
    bp_values,
    size=800,
    replace=True
)

bootstrap_mean = bootstrap_sample.mean()
print(bootstrap_mean)
# %%
bootstrap_means = []

for i in range(1000):
    bootstrap_sample = np.random.choice(
        bp_values,
        size=800,
        replace=True
    )

    bootstrap_mean = bootstrap_sample.mean()

    bootstrap_means.append(bootstrap_mean)

    print(len(bootstrap_means))
# %%
bootstrap_ci = np.percentile(bootstrap_means, [2.5, 97.5])

print(bootstrap_ci)
# %%
print(df.columns.tolist())
# %%
print(df["smoker"].value_counts())

# %%
df["smoker"] = df["smoker"].str.strip()
# %%
bp_smokers = df[df["smoker"] == "Yes"]["systolic_bp"]

bp_non_smokers = df[df["smoker"] == "No"]["systolic_bp"]

print(bp_smokers.mean())
print(bp_non_smokers.mean())
# %%
from scipy import stats

# %%
t_stat, p_value = stats.ttest_ind(
    bp_smokers,
    bp_non_smokers,
    equal_var=False,
    alternative="greater"
)

print("t-värde:", t_stat)
print("p-värde:", p_value)

# %%
np.random.seed(42)

non_smokers_sim = np.random.normal(
    loc=149,
    scale=13,
    size=100
)

smokers_sim = np.random.normal(
    loc=152,
    scale=13,
    size=100
)
# %%
print(non_smokers_sim.mean())
print(smokers_sim.mean())

# %%
print(non_smokers_sim.std())
print(smokers_sim.std())

# %%
t_stat, p_value = stats.ttest_ind(
    smokers_sim,
    non_smokers_sim,
    equal_var=False,
    alternative="greater"
)

print("p-värde:", p_value)

# %%
significant_results = []

for i in range(1000):

    non_smokers_sim = np.random.normal(
        loc=149,
        scale=13,
        size=100
    )

    smokers_sim = np.random.normal(
        loc=154,
        scale=13,
        size=100
    )

    t_stat, p_value = stats.ttest_ind(
        smokers_sim,
        non_smokers_sim,
        equal_var=False,
        alternative="greater"
    )

    significant_results.append(p_value < 0.05)

power = significant_results.count(True) / len(significant_results)

print("Statistical power:", power)
print("Power in percent:", power * 100)

# %%
power_results = pd.DataFrame({
    "sample_size": [100, 200, 300, 100],
    "mean_difference": [3, 3, 3, 5],
    "power": [0.476, 0.735, 0.897, 0.859]
})

power_results
# %%
power_by_sample = power_results[
    power_results["mean_difference"] == 3
]

plt.figure(figsize=(8, 5))

plt.plot(
    power_by_sample["sample_size"],
    power_by_sample["power"],
    marker="o"
)

plt.axhline(
    y=0.80,
    linestyle="--",
    label="80 % power"
)

plt.xlabel("Antal personer per grupp")
plt.ylabel("Statistical power")
plt.title("Effect of sample size on statistical power")
plt.legend()
plt.grid()

plt.show()

# %%
power_by_difference = power_results[
    power_results["sample_size"] == 100
]

plt.figure(figsize=(8, 5))

plt.bar(
    power_by_difference["mean_difference"],
    power_by_difference["power"]
)

plt.axhline(y=0.80, linestyle="--", label="80 % power")

plt.xlabel("Difference between group means (mmHg)")
plt.ylabel("Statistical power")
plt.title("Effect of mean difference on statistical power")
plt.legend()

plt.show()
# %%
