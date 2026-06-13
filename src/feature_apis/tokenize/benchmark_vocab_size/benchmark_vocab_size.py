
import os
from pathlib import Path

from project_context.project_context import ProjectContext
from feature_apis.tokenize.sp_tokenize import (
    create_all_text_file,
    round_rows,
    save_tsv,
    tokenize,
)

"""
プロジェクトのカレントディレクトリから実行する前提です
本ファイルまで cd する必要はありません
"""

OUTPUT_SUBDIR = f"{Path(__file__).parent.name}/{Path(__file__).stem}"

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