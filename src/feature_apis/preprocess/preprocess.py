
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

def load_vocab(ctx: ProjectContext):
    db_file = ctx.normalized_db_file
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("SELECT sentence FROM chABSA")
        sentences = [row[0] for row in c.fetchall()]
    vocab = set()
    for sentence in sentences:
        for word in sentence.split():
            vocab.add(word)
    return vocab