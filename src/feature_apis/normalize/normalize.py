import os
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

from project_context.project_context import ProjectContext

OUTPUT_SUBDIR = f"{Path(__file__).parent.name}/{Path(__file__).stem}"

def unicode_normalize(s):
    # Unicode正規化を行う
    # 例: "ＡＢＣａｂｃ１２３" -> "ＡＢＣａｂｃ１２３" (全角英数字はそのまま、半角スペースは全角スペースに変換される)
    return unicodedata.normalize("NFKC", s)

def to_full_width(s):
    return s.translate(str.maketrans(
        {chr(i): chr(i + 0xFEE0) for i in range(0x21, 0x7F)}
    ))

def pre_normalize(s):
    s = unicode_normalize(s)
    s = to_full_width(s)
    return s

##############################
# 以下の処理は、全て全角統一後に行うことを想定している
##############################


def unify_brackets(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提
    # 始まり括弧として扱う文字
    open_brackets = (
        "（"
        "［"
        "｛"
        "〈"
        "《"
        "「"
        "『"
        "【"
        "〔"
        "〖"
        "〘"
        "〚"
        "〝"
        "‘"
        "“"
        "｟"
        "〿"
        "⦅"
        "⦃"
        "⟨"
        "⟪"
        "⦇"
        "⟦"
        "⟬"
        "⟮"
        "⦉"
        "⦋"
        "⦍"
        "⦏"
        "⦑"
        "⦓"
        "⦕"
        "⦗"
    )

    # 終わり括弧として扱う文字
    close_brackets = (
        "）"
        "］"
        "｝"
        "〉"
        "》"
        "」"
        "』"
        "】"
        "〕"
        "〗"
        "〙"
        "〛"
        "〟"
        "’"
        "”"
        "｠"
        "⦆"
        "⦄"
        "⟩"
        "⟫"
        "⦈"
        "⟧"
        "⟭"
        "⟯"
        "⦊"
        "⦌"
        "⦎"
        "⦐"
        "⦒"
        "⦔"
        "⦖"
        "⦘"
    )

    open_pattern = "[" + re.escape(open_brackets) + "]"
    close_pattern = "[" + re.escape(close_brackets) + "]"

    s = re.sub(open_pattern, "（", s)
    s = re.sub(close_pattern, "）", s)

    return s

def unify_horizontal_bars(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提

    horizontal_bars = (
        "－"  # fullwidth hyphen-minus
        "−"  # minus sign
        "‐"  # hyphen
        "-"  # non-breaking hyphen
        "‒"  # figure dash
        "–"  # en dash
        "—"  # em dash
        "―"  # horizontal bar
        "⁃"  # hyphen bullet
        "﹘"  # small em dash
        "﹣"  # small hyphen-minus
        "ー"  # Japanese prolonged sound mark
        "ｰ"  # halfwidth katakana-hiragana prolonged sound mark
        "─"  # box drawings light horizontal
        "━"  # box drawings heavy horizontal
        "╌"  # box drawings light double dash horizontal
        "╍"  # box drawings heavy double dash horizontal
        "┄"  # box drawings light triple dash horizontal
        "┅"  # box drawings heavy triple dash horizontal
        "┈"  # box drawings light quadruple dash horizontal
        "┉"  # box drawings heavy quadruple dash horizontal
        "╴"  # box drawings light left
        "╶"  # box drawings light right
        "╸"  # box drawings heavy left
        "╺"  # box drawings heavy right
    )

    pattern = "[" + re.escape(horizontal_bars) + "]"
    return re.sub(pattern, "－", s)

def unify_alphabet_case(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提
    # 全角英小文字を全角英大文字に統一する

    lower_alphabets = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    upper_alphabets = "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"

    translation_table = str.maketrans(lower_alphabets, upper_alphabets)

    return s.translate(translation_table)

def unify_commas(s: str) -> str:
    # 読点・カンマっぽいものを「、」に統一する

    comma_chars = (
        "、"
        "，"
        ","
        "､"
        "﹐"
        "﹑"
    )

    pattern = "[" + re.escape(comma_chars) + "]"
    return re.sub(pattern, "、", s)


def unify_periods(s: str) -> str:
    # 句点・ピリオドっぽいものを「．」に統一する

    period_chars = (
        "。"
        "．"
        "."
        "｡"
        "﹒"
    )

    pattern = "[" + re.escape(period_chars) + "]"
    return re.sub(pattern, "．", s)

def unify_slashes(s: str) -> str:
    slash_chars = (
        "／"
        "/"
        "＼"
        "\\"
        "⁄"
        "∕"
    )

    pattern = "[" + re.escape(slash_chars) + "]"
    return re.sub(pattern, "／", s)

def unify_waves(s: str) -> str:
    wave_chars = (
        "〜"
        "～"
        "~"
        "∼"
        "∽"
        "∿"
    )

    pattern = "[" + re.escape(wave_chars) + "]"
    return re.sub(pattern, "～", s)

def unify_quotes(s: str) -> str:
    open_quotes = (
        "“"
        "‘"
        "〝"
        "＂"
        "\""
        "'"
    )

    close_quotes = (
        "”"
        "’"
        "〟"
        "＂"
        "\""
        "'"
    )

    s = re.sub("[" + re.escape(open_quotes) + "]", "「", s)
    s = re.sub("[" + re.escape(close_quotes) + "]", "」", s)

    return s

def create_normalized_db(ctx: ProjectContext):
    # normalized_db_file の削除
    normalized_db_file = ctx.normalized_db_file
    if os.path.exists(normalized_db_file):
        ctx.info(f"Removing existing normalized DB file: {normalized_db_file}")
        os.remove(normalized_db_file)

    # row_id(int), polarity(string), target(string), sentence(string) の順で保存する
    # テーブル作成
    ctx.info("Creating normalized SQLite DB and table...")
    conn = sqlite3.connect(normalized_db_file)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chABSA (
                    row_id INTEGER PRIMARY KEY, 
                    polarity TEXT,
                    target TEXT,
                    sentence TEXT
                )''')
    conn.commit()
    conn.close()

def unify_digits_012(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提
    # [0]     → ０
    # [1, 2]  → １
    # [!012]  → ３

    digit_table = str.maketrans({
        "０": "０",
        "１": "１",
        "２": "１",
        "３": "３",
        "４": "３",
        "５": "３",
        "６": "３",
        "７": "３",
        "８": "３",
        "９": "３",
    })

    return s.translate(digit_table)

def normalize(ctx: ProjectContext, sentence: str) -> str:
    sentence = pre_normalize(sentence)
    sentence = unify_brackets(sentence)
    sentence = unify_horizontal_bars(sentence)
    sentence = unify_alphabet_case(sentence)
    sentence = unify_commas(sentence)
    sentence = unify_periods(sentence)
    sentence = unify_slashes(sentence)
    sentence = unify_waves(sentence)
    sentence = unify_quotes(sentence)
    sentence = unify_digits_012(sentence)

    return sentence

def execute_normalize(ctx: ProjectContext):
    ctx.info("Starting normalization process...")
    create_normalized_db(ctx)

    # raw_db_file を読み込んで正規化後に normalized_db_file に保存する
    with sqlite3.connect(ctx.raw_db_file) as read_conn:
        # 件数を取得、進捗表示する
        read_c = read_conn.cursor()
        read_c.execute("SELECT COUNT(*) FROM chABSA")
        total_rows = read_c.fetchone()[0]
        ctx.info(f"Total rows to normalize: {total_rows}")
        with sqlite3.connect(ctx.normalized_db_file) as write_conn:
            read_c = read_conn.cursor()
            write_c = write_conn.cursor()

            read_c.execute("SELECT row_id, polarity, target, sentence FROM chABSA")
            rows = read_c.fetchall()

            for irow, row in enumerate(rows, start=1):
                row_id, polarity, target, sentence = row
                normalized_sentence = pre_normalize(sentence)
                normalized_sentence = normalize(ctx, normalized_sentence)

                write_c.execute(
                    "INSERT INTO chABSA (row_id, polarity, target, sentence) VALUES (?, ?, ?, ?)",
                    (row_id, polarity, target, normalized_sentence)
                )
                if irow % 500 == 0:
                    ctx.info(f"Normalized row {irow}/{total_rows}")
    ctx.info("Finished normalization process.")

def execute_view_samples(ctx: ProjectContext, num_samples=5):
    with sqlite3.connect(ctx.raw_db_file) as raw_conn:
        raw_c = raw_conn.cursor()
        raw_c.execute(
            """
            SELECT row_id, polarity, target, sentence
            FROM chABSA
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (num_samples,)
        )
        raw_rows = raw_c.fetchall()

    if not raw_rows:
        ctx.info("No samples found in raw DB.")
        return

    row_ids = [row[0] for row in raw_rows]
    placeholders = ",".join("?" for _ in row_ids)

    with sqlite3.connect(ctx.normalized_db_file) as normalized_conn:
        normalized_c = normalized_conn.cursor()
        normalized_c.execute(
            f"""
            SELECT row_id, polarity, target, sentence
            FROM chABSA
            WHERE row_id IN ({placeholders})
            """,
            row_ids,
        )
        normalized_rows = normalized_c.fetchall()

    normalized_row_by_id = {row[0]: row for row in normalized_rows}

    for row_id, polarity, target, sentence_before in raw_rows:
        normalized_row = normalized_row_by_id.get(row_id)
        if normalized_row is None:
            ctx.warn(f"Normalized row not found for row_id={row_id}")
            continue

        sentence_after = normalized_row[3]

        print("ーーーーー")
        print("ヘッダ情報")
        print(f"Row ID: {row_id}")
        print(f"Polarity: {polarity}")
        print(f"Target: {target}")
        print("ーーーーー")
        print("正規化前")
        print(sentence_before)
        print("正規化後")
        print(sentence_after)

def export_samples(ctx: ProjectContext, num_samples=5, out_file="normalized_samples.tsv"):
    # TSV で出力する
    out_path = os.path.join(ctx.data_dir, OUTPUT_SUBDIR, out_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with sqlite3.connect(ctx.normalized_db_file) as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT row_id, polarity, target, sentence
            FROM chABSA
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (num_samples,)
        )
        rows = c.fetchall()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("row_id\tpolarity\ttarget\tsentence\n")
        for row in rows:
            row_id, polarity, target, sentence = row
            f.write(f"{row_id}\t{polarity}\t{target}\t{sentence}\n")
    ctx.info(f"Exported {num_samples} normalized samples to {out_path}")

def main():
    ctx = ProjectContext()
    function_name = sys.argv[1]
    if function_name == "normalize":
        execute_normalize(ctx)
    elif function_name == "view_samples":
        execute_view_samples(ctx)
    elif function_name == "export_samples":
        export_samples(ctx)
    else:
        print(f"Unknown function name: {function_name}")


if __name__ == "__main__":
    main()

"""
python src/feature_apis/normalize/normalize.py normalize
python src/feature_apis/normalize/normalize.py view_samples
python src/feature_apis/normalize/normalize.py export_samples
"""
