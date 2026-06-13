from feature_apis.preprocess.preprocess import (
    VocabContext,
    ValidRows,
    create_vocab_context,
    create_vocab_indexes,
    load_spm_model,
    load_xy,
)
from project_context.project_context import ProjectContext


def generate_one_hot_sum_vector(vctx: VocabContext, sentence: str):
    """1 文書の BoW を dense ベクトル化したカウントベクトルを返す。"""
    token_ids = create_vocab_indexes(vctx.spm, sentence)
    one_hot_sum_vector = [0] * vctx.size
    for token_id in token_ids:
        if token_id < vctx.size:
            one_hot_sum_vector[token_id] += 1
    return one_hot_sum_vector


def create_document_bow(x, y, spm):
    """BoW 行列（文書数 x 語彙数）を作成する。"""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    vctx = create_vocab_context(spm)
    bow_matrix = []

    for sentence in x:
        bow_row = generate_one_hot_sum_vector(vctx, sentence)
        bow_matrix.append(bow_row)

    return bow_matrix


def create_document_bow_from_ctx(ctx: ProjectContext, valid_rows: ValidRows):
    """ProjectContext から row_ids/x/y/spm を読み込み BoW 行列を作成する。"""
    row_ids, x, y = load_xy(ctx, valid_rows)
    spm = load_spm_model(ctx)
    bow_matrix = create_document_bow(x, y, spm)
    return row_ids, bow_matrix


def count_word_frequencies(row_ids, x, y, spm):
    """コーパス全体で token_id の総出現回数を数える。"""
    bow_matrix = create_document_bow(x, y, spm)
    if not bow_matrix:
        return {}

    vocab_size = len(bow_matrix[0])
    word_frequencies = {token_id: 0 for token_id in range(vocab_size)}

    for row in bow_matrix:
        for token_id, count in enumerate(row):
            word_frequencies[token_id] += count

    return {
        token_id: count
        for token_id, count in word_frequencies.items()
        if count > 0
    }


def create_bow_vectors(row_ids, x, y, spm):
    """BoW 行列を作成する（create_document_bow の別名）。"""
    return create_document_bow(x, y, spm)
