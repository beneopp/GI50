import pandas as pd
import numpy as np

f = "../data/GI50/DTP_NCI60_RAW.csv"
df = pd.read_csv(f, header=8, na_values="na")

index = df["NSC #"]
df = df.iloc[:, 9:]
df.index = index

df = df.groupby("NSC #").agg("mean")

drug_f = "../data/drugs/fingerprints.csv"
drug_df = pd.read_csv(drug_f)
del drug_f
drug_df.index = drug_df["NSC #"]
drug_df = drug_df.iloc[:, 1:]

identified_drugs = np.intersect1d(drug_df.index, df.index)
df = df.loc[identified_drugs]
del identified_drugs

genetic_f = "../data/cell_line/RNA__Affy_HuEx_1.0_GCRMA.txt.txt"
genetic_df = pd.read_csv(genetic_f, sep="\t", na_values="-")
del genetic_f

genetic_df = genetic_df.iloc[:, 7:]
genetic_df = genetic_df.dropna(axis=1, how="all")

cell_lines = genetic_df.columns
df = df[cell_lines]
del cell_lines

df["NSC #"] = df.index
df = pd.melt(df, id_vars=["NSC #"], value_vars=df.columns[df.columns != "NSC #"])
df = df.rename(columns={"variable":"Cell Line", "value":"GI50 Value"})

df = df.dropna(axis=0)

drug_df = drug_df.add_prefix("drug_fingerprint_")
df = pd.merge(df, drug_df, left_on="NSC #", right_on=drug_df.index)
del drug_df

genetic_df = genetic_df.transpose()
genetic_df = genetic_df.add_prefix("genetic_expr_")

print("merging")
df = pd.merge(df, genetic_df, left_on="Cell Line", right_on=genetic_df.index)
print(df)
