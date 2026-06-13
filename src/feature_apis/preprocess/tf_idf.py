import math

from feature_apis.preprocess.bow import create_document_bow
from feature_apis.preprocess.preprocess import ValidRows, load_spm_model, load_xy
from project_context.project_context import ProjectContext

# TF-IDF の計算に必要な関数を定義するファイル


def count_document_frequencies(bow_matrix: list):
    """DF（token を含む文書数）を数える。"""
    if not bow_matrix:
        return []

    vocab_size = len(bow_matrix[0])
    document_frequencies = [0] * vocab_size

    for bow_row in bow_matrix:
        for token_id, count in enumerate(bow_row):
            if count > 0:
                document_frequencies[token_id] += 1

    return document_frequencies


def calculate_idf_values(document_frequencies: list, num_documents: int):
    """スムージング付き IDF を計算する。idf = log((N+1)/(df+1)) + 1"""
    idf_values = []
    for df in document_frequencies:
        idf_values.append(math.log((num_documents + 1) / (df + 1)) + 1.0)
    return idf_values


def convert_bow_to_tfidf(bow_row: list, idf_values: list):
    """1 文書分の BoW 行ベクトルを TF-IDF 行ベクトルに変換する。"""
    total_tokens = sum(bow_row)
    if total_tokens == 0:
        return [0.0] * len(bow_row)

    tfidf_row = [0.0] * len(bow_row)
    for token_id, count in enumerate(bow_row):
        if count == 0:
            continue
        tf = count / total_tokens
        idf = idf_values[token_id] if token_id < len(idf_values) else 0.0
        tfidf_row[token_id] = tf * idf

    return tfidf_row


def create_tfidf_vectors(row_ids, x, y, spm):
    """
    row_ids/x/y/spm から TF-IDF を作る。
    戻り値:
      - tfidf_vectors: {row_id: [tfidf, ...]}
      - idf_values: [idf, ...]
    """
    bow_matrix = create_document_bow(x, y, spm)
    num_documents = len(bow_matrix)

    document_frequencies = count_document_frequencies(bow_matrix)
    idf_values = calculate_idf_values(document_frequencies, num_documents)

    tfidf_vectors = {}
    for row_id, bow_row in zip(row_ids, bow_matrix):
        tfidf_vectors[row_id] = convert_bow_to_tfidf(bow_row, idf_values)

    return tfidf_vectors, idf_values


def create_tfidf_vectors_from_ctx(ctx: ProjectContext, valid_rows: ValidRows):
    """ProjectContext から読み込んだデータを使って TF-IDF を作成する。"""
    row_ids, x, y = load_xy(ctx, valid_rows)
    spm = load_spm_model(ctx)
    return create_tfidf_vectors(row_ids, x, y, spm)
