import sentencepiece
from project_context.project_context import ProjectContext
import sqlite3
from feature_apis.normalize.normalize import normalized_db_file_name

class VocabContext:
    def __init__(self, spm, size, i2w, w2i):
        self.spm = spm
        self.size = size
        self.index_to_word = i2w # インデックスから単語への辞書
        self.word_to_index = w2i # 単語からインデックスへの辞書

class ValidRows:
    def __init__(self, valid_row_ids: list[int]):
        self.filtered = len(valid_row_ids) > 0
        self.valid_row_ids = valid_row_ids

def load_xy(ctx: ProjectContext, valid_rows: ValidRows):
    db_file = normalized_db_file_name(ctx)
    rows = []
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("SELECT row_id, polarity, sentence FROM chABSA")
        for row in c.fetchall():
            row_id = row[0]
            if valid_rows.filtered and row_id not in valid_rows.valid_row_ids:
                continue
            rows.append(row)
    x = [row[2] for row in rows]
    y = [row[1] for row in rows]
    row_ids = [row[0] for row in rows]
    return row_ids, x, y

def load_spm_model(ctx: ProjectContext):
    spm_model = sentencepiece.SentencePieceProcessor()
    spm_model.load(ctx.vocab_model_file)
    return spm_model

def load_spm_vocab(ctx: ProjectContext):
    with open(ctx.vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f.readlines()]
    return vocab

def load_spm(ctx: ProjectContext):
    spm_model = load_spm_model(ctx)
    vocab = load_spm_vocab(ctx)
    return spm_model, vocab

# 単語数（サイズ）を取得する
def get_vocab_size(spm) -> int:
    return spm.get_piece_size()

# インデックス, 対応単語の辞書型を取得する
def get_index_to_word_dict(spm) -> dict:
    index_to_word_dict = {i: spm.IdToPiece(i) for i in range(spm.get_piece_size())}
    return index_to_word_dict

# 単語、インデックスの辞書型を取得する
def get_word_to_index_dict(spm) -> dict:
    word_to_index_dict = {spm.IdToPiece(i): i for i in range(spm.get_piece_size())}
    return word_to_index_dict

# 単語をインデックスに変換する関数
def create_vocab_indexes(spm, sentence):
    return spm.EncodeAsIds(sentence)

def create_vocab_context(spm) -> VocabContext:
    vocab_size = get_vocab_size(spm)
    index_to_word_dict = get_index_to_word_dict(spm)
    word_to_index_dict = get_word_to_index_dict(spm)
    vocab_context = VocabContext(
        spm=spm,
        size=vocab_size,
        i2w=index_to_word_dict,
        w2i=word_to_index_dict
    )
    return vocab_context
