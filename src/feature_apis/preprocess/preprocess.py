
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