import sentencepiece
from project_context.project_context import ProjectContext
import sqlite3

def load_xy(ctx: ProjectContext):
    db_file = ctx.normalized_db_file
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("SELECT polarity, sentence FROM chABSA")
        data = c.fetchall()
    X = [row[1] for row in data]
    y = [row[0] for row in data]
    return X, y

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


