import os

DIR = "../datasets/translated"

LC = 0

for file in os.listdir(DIR):
    nf = file
    nf = nf.replace("part", "").replace("_en_to_kha.txt", "")
    nf = int(nf)
    with open(f"{DIR}/{file}", "r", encoding="utf-8") as fp:
        LC += len(fp.readlines())

print(f"Lines translated : {LC}/{5000000}")
