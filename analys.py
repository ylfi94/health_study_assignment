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
