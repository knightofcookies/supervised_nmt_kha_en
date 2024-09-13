import os

DIR = "../datasets/translated"

LC = 0

for file in os.listdir(DIR):
    with open(f"{DIR}/{file}", "r", encoding="utf-8") as fp:
        LC += len(fp.readlines())

print(f"Lines translated : {LC}/{1000000}")
