
import os
import sqlite3
from collections import Counter
from pathlib import Path

import sentencepiece

from project_context.project_context import ProjectContext

"""
プロジェクトのカレントディレクトリから実行する前提です
本ファイルまで cd する必要はありません
"""

OUTPUT_SUBDIR = f"{Path(__file__).parent.name}/{Path(__file__).stem}"

def create_all_text_file(ctx: ProjectContext, out_dir: str):
    # db_file = ctx.raw_db_file
    db_file = ctx.normalized_db_file
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("SELECT sentence FROM chABSA")
        sentences = [row[0] for row in c.fetchall()]
    sentences_file = os.path.join(out_dir, "sentences.txt")
    os.makedirs(os.path.dirname(sentences_file), exist_ok=True)
    with open(sentences_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")
    return sentences_file

"""
# row_id(int), polarity(string), target(string), sentence(string)
# SELECT row_id, polarity, target, sentence FROM chABSA
"""
def tokenize(ctx: ProjectContext, sentences_file: str, vocab_size: int):
    ctx.info(f"Training SentencePiece model with vocab size {vocab_size}...")

    model_prefix = os.path.join(
        os.path.dirname(sentences_file),
        f"spm_{vocab_size}"
    )

    sentencepiece.SentencePieceTrainer.train(
        input=sentences_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    model_file = f"{model_prefix}.model"

    sp = sentencepiece.SentencePieceProcessor()
    sp.load(model_file)

    with open(sentences_file, "r", encoding="utf-8") as f:
        sentences = [line.rstrip("\n") for line in f if line.strip()]

    sentence_count = len(sentences)

    token_count = 0
    char_count = 0
    unk_count = 0
    used_piece_ids_counter = Counter()

    unk_id = sp.unk_id()

    for sentence in sentences:
        ids = sp.encode(sentence, out_type=int)

        token_count += len(ids)
        char_count += len(sentence)
        unk_count += sum(1 for token_id in ids if token_id == unk_id)
        used_piece_ids_counter.update(ids)

    vocab_actual_size = sp.get_piece_size()
    used_pieces = len(used_piece_ids_counter)

    special_ids = {
        sp.pad_id(),
        sp.unk_id(),
        sp.bos_id(),
        sp.eos_id(),
    }

    used_special_pieces = sum(
        1 for piece_id in used_piece_ids_counter if piece_id in special_ids
    )

    used_pieces_excluding_special = used_pieces - used_special_pieces

    avg_token_per_sentence = (
        token_count / sentence_count if sentence_count > 0 else 0
    )

    avg_token_per_char = (
        token_count / char_count if char_count > 0 else 0
    )

    unk_rate = (
        unk_count / token_count if token_count > 0 else 0
    )

    used_vocab_rate = (
        used_pieces / vocab_actual_size if vocab_actual_size > 0 else 0
    )

    ctx.info(
        f"Vocab size: {vocab_size}, "
        f"Actual vocab size: {vocab_actual_size}, "
        f"Token count: {token_count}, "
        f"Char count: {char_count}, "
        f"Sentence count: {sentence_count}, "
        f"UNK count: {unk_count}, "
        f"UNK rate: {unk_rate:.4f}, "
        f"Used pieces: {used_pieces}, "
        f"Used pieces excluding special tokens: {used_pieces_excluding_special}, "
        f"Used vocab rate: {used_vocab_rate:.4f}, "
        f"Avg token/sentence: {avg_token_per_sentence:.2f}, "
        f"Avg token/char: {avg_token_per_char:.4f}"
    )

    return [
        vocab_actual_size,
        token_count,
        char_count,
        sentence_count,
        unk_count,
        unk_rate,
        used_pieces,
        used_pieces_excluding_special,
        used_vocab_rate,
        avg_token_per_sentence,
        avg_token_per_char,
    ]

def round_f3(x):
    return round(x, 3)

def round_row(row):
    return [round_f3(x) if isinstance(x, float) else x for x in row]

def round_rows(rows):
    return [round_row(row) for row in rows]

def save_tsv(header, rows, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            if len(row) != len(header):
                raise ValueError(
                    f"Header has {len(header)} columns, but row has {len(row)} columns: {row}"
                )
            row_str = "\t".join(str(x) for x in row)
            f.write(row_str + "\n")

def main():
    ctx = ProjectContext()
    ctx.info("Starting benchmark for vocab size...")
    out_dir = os.path.join(ctx.data_dir, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    sentences_file = create_all_text_file(ctx, out_dir)
    # size_list = [100, 200, 400, 800, 1600]
    minimum = 1359 # 少なすぎるとエラーになるので、エラー時に出てきた最小数を指定しておく
    size_list = [
        0 + minimum,
        100 + minimum,
        200 + minimum,
        400 + minimum,
        800 + minimum,
        1600 + minimum,
        3200 + minimum,
        6400 + minimum,
    ]

    sum_rows = []
    for size in size_list:
        ctx.info(f"Tokenizing with vocab size {size}...")
        row = tokenize(ctx, sentences_file, size)
        sum_rows.append(row)

    ctx.info("Finished benchmark for vocab size.")

    # show results
    ctx.info("Results:")
    header = "Vocab Size, Actual Vocab Size, Token Count, Char Count, Sentence Count, UNK Count, UNK Rate, Used Pieces, Used Pieces Excl. Special Tokens, Used Vocab Rate, Avg Token/Sentence, Avg Token/Char"
    ctx.info(header)
    rounded_rows = round_rows([
        [size, *row] for size, row in zip(size_list, sum_rows)
    ])
    for row in rounded_rows:
        row_str = ", ".join(str(x) for x in row)
        ctx.info(row_str)

    save_tsv(
        header.split(", "),
        rounded_rows,
        os.path.join(out_dir, "benchmark_results.tsv")
    )

if __name__ == "__main__":
    main()

"""
python src/feature_apis/tokenize/benchmark_vocab_size/benchmark_vocab_size.py
"""