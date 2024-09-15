import winsound
import datetime
import os
import asyncio
import pickle
import concurrent.futures
from typing import List
from deep_translator import GoogleTranslator
from win11toast import toast

# Add {"khasi": "kha"} to the Google Translate dict in constants.py in deep_translator

THREAD_LIMIT = 250

START = 1  # Delete pickle_dump if you change this
END = 800  # Delete pickle_dump if you change this

PICKLE_DUMP_PATH = "pickle_dump"

if not os.path.exists(PICKLE_DUMP_PATH):
    complete: List[bool] = [False] * (END - START + 1)
    with open(PICKLE_DUMP_PATH, "wb") as fp:
        pickle.dump(complete, fp)

with open(PICKLE_DUMP_PATH, "rb") as fp:
    complete = pickle.load(fp)


def translate_chunk(chunk: str, complete_index: int) -> None:

    source = "en"
    target = "kha"

    with open(f"../datasets/parts/{chunk}.txt", "r", encoding="utf-8") as cfp:
        orig_lines = cfp.readlines()

    index = 0
    file_path = f"../datasets/translated/{chunk}_{source}_to_{target}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as cfp:
            index = len(cfp.readlines())
    if index < 0:
        index = 0

    if not os.path.exists("../datasets/translated/"):
        os.mkdir("../datasets/translated")

    translator = GoogleTranslator(source=source, target=target)

    while index < len(orig_lines):
        line = orig_lines[index]
        if len(line) < 5000:
            txt = translator.translate(line)
        else:
            txt = translator.translate(line[:4999])
        if txt is not None:
            with open(file_path, "a", encoding="utf-8") as cfp:
                cfp.write(txt + "\n")
        index += 1

    complete[complete_index] = True


async def main() -> None:

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_LIMIT)

    chunk_index = 0
    while chunk_index < (END - START + 1):
        if not complete[chunk_index]:
            pool.submit(translate_chunk, f"part{START+chunk_index}", chunk_index)
        chunk_index += 1

    pool.shutdown(wait=True)


if __name__ == "__main__":
    start = datetime.datetime.now()
    asyncio.run(main())
    end = datetime.datetime.now()
    print(end - start)
    winsound.Beep(500, 500)
    with open(PICKLE_DUMP_PATH, "wb") as f:
        pickle.dump(complete, f)
    incomplete: List[int] = []
    COUNT = 0
    for i in range(0, END - START + 1):
        if complete[i]:
            COUNT += 1
        else:
            incomplete.append(START + i)
    if COUNT == len(complete):
        toast(
            "Program Terminated",
            f"{COUNT}/{len(complete)} chunks in range successfully translated.",
        )
    else:
        toast(
            "Program Terminated",
            f"""{COUNT}/{len(complete)} chunks in range successfully translated. 
Change your IP address to finish the rest.""",
        )
        print("Translation pending for chunks: ", incomplete)
