import pandas as pd

df = pd.read_csv("../datasets/kha.csv", sep="\t")

en_lines = df["English"].values
kha_lines = df["Khasi"].values

nel = []
nkl = []

for l in en_lines:
    nel.append(l + "\n")

for l in kha_lines:
    nkl.append(l + "\n")

with open("en_nits.txt", "w", encoding="utf-8") as fp:
    fp.writelines(nel)

with open("kha_nits.txt", "w", encoding="utf-8") as fp:
    fp.writelines(nkl)
