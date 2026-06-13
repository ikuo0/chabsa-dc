import json
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path

from datasets import load_dataset

from project_context.project_context import ProjectContext

OUTPUT_SUBDIR = f"{Path(__file__).parent.name}/{Path(__file__).stem}"

def raw_db_file_name(ctx: ProjectContext):
    return os.path.join(ctx.data_dir, OUTPUT_SUBDIR, "raw_dataset.db")

def download(ctx: ProjectContext):
    ds = load_dataset("TheFinAI/jp-chABSA")

    # 一旦pickleで全て保存
    pickle_file = os.path.join(ctx.data_dir, OUTPUT_SUBDIR, "dataset.pkl")
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    with open(pickle_file, "wb") as f:
        pickle.dump(ds, f)

##############################
# キーを確認する
##############################
def enum_keys(ctx: ProjectContext):
    pickle_file = os.path.join(ctx.data_dir, OUTPUT_SUBDIR, "dataset.pkl")
    with open(pickle_file, "rb") as f:
        ds = pickle.load(f)

    print(ds)
    print(ds["train"])
    print(ds["train"].features)
    parent_keys = ds.keys()
    print(f"Parent keys: {parent_keys}")
    for i in range(3):
        print(f"Example {i}:")
        print(ds["train"][i])

###############################
# データセットを分割して SQLite に保存する
###############################
def split_save(ctx: ProjectContext):
    # enum_keys でキーを確認してから都合の良いファイル分けで保存する

    # DBファイルを削除
    raw_db_file = raw_db_file_name(ctx)
    if os.path.exists(raw_db_file):
        ctx.info(f"Removing existing SQLite DB file: {raw_db_file}")
        os.remove(raw_db_file)

    # データセットを分割して保存
    # row_id(int), polarity(string), target(string), sentence(string) の順で保存する
    # テーブル作成
    ctx.info("Creating SQLite DB and table...")
    conn = sqlite3.connect(raw_db_file)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chABSA (
                    row_id INTEGER PRIMARY KEY, 
                    polarity TEXT,
                    target TEXT,
                    sentence TEXT
                )''')
    conn.commit()
    conn.close()

    pickle_file = os.path.join(ctx.data_dir, OUTPUT_SUBDIR, "dataset.pkl")
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    with open(pickle_file, "rb") as f:
        ds = pickle.load(f)
    num_rows = ds["train"].num_rows
    ctx.info(f"Number of rows in dataset: {num_rows}")
    row_id = 1 # 加算してキーとする
    # conn = sqlite3.connect(raw_db_file)
    with sqlite3.connect(raw_db_file_name(ctx)) as conn:
        for i in range(num_rows):
            polarity = ds["train"][i]["polarity"]
            target = ds["train"][i]["target"]
            sentence = ds["train"][i]["sentence"]

            # DBに保存
            c = conn.cursor()
            c.execute("INSERT INTO chABSA (row_id, polarity, target, sentence) VALUES (?, ?, ?, ?)", 
                    (row_id, polarity, target, sentence))
            row_id += 1
            if (i%500 == 0):
                ctx.info(f"Inserted {i} rows...")
    ctx.info(f"Finished inserting {num_rows} rows into SQLite DB: {raw_db_file_name(ctx)}")

def view_samples(ctx: ProjectContext, num_samples=5):
    with sqlite3.connect(raw_db_file_name(ctx)) as conn:
        c = conn.cursor()
        c.execute("SELECT row_id, polarity, target, sentence FROM chABSA LIMIT ?", (num_samples,))
        rows = c.fetchall()
        for row in rows:
            print(f"Row ID: {row[0]}, Polarity: {row[1]}, Target: {row[2]}, Sentence: {row[3]}")

def main():
    ctx = ProjectContext()
    function_name = sys.argv[1]
    if function_name == "download":
        download(ctx)
    elif function_name == "enum_keys":
        enum_keys(ctx)
    elif function_name == "split_save":
        split_save(ctx)
    elif function_name == "sample":
        view_samples(ctx, 5)
    else:
        print(f"Unknown function name: {function_name}")    

main()

"""
python src/feature_apis/dataset/download_chABSA.py download
python src/feature_apis/dataset/download_chABSA.py enum_keys
python src/feature_apis/dataset/download_chABSA.py split_save
python src/feature_apis/dataset/download_chABSA.py sample
"""