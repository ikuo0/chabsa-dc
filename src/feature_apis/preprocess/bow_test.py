from feature_apis.preprocess.preprocess import VocabContext, create_vocab_context, create_vocab_indexes, ValidRows, load_xy, load_spm_model
from feature_apis.preprocess.bow import create_document_bow
from feature_apis.preprocess.tf_idf import count_document_frequencies, calculate_idf_values, convert_bow_to_tfidf
from project_context.project_context import ProjectContext

VALID_ROWS = ValidRows([1, 2, 3])


"""
pytest -s -vv src/feature_apis/preprocess/bow_test.py::test_create_document_bow
"""
def test_create_document_bow():
    ctx = ProjectContext()
    valid_rows = VALID_ROWS
    row_ids, x, y = load_xy(ctx, valid_rows)
    spm = load_spm_model(ctx)
    bow_matrix = create_document_bow(x, y, spm)
    assert len(row_ids) == len(bow_matrix)
    assert len(row_ids) == len(valid_rows.valid_row_ids)
    # print
    for row_id, bow_row in zip(row_ids, bow_matrix):
        print(f"row_id: {row_id}, bow_row: {bow_row}")


"""
pytest -s -vv src/feature_apis/preprocess/bow_test.py::test_create_onehot
"""
def test_create_onehot():
    """
    data/tokenize/sp_tokenize/spm.vocab を見て手動で設定し想定通りのインデックスに１が加算されることを確認する
8 百万円
22 前期比
86 個人消費
160 設備投資
    """
    ctx = ProjectContext()
    spm = load_spm_model(ctx)
    vctx = create_vocab_context(spm)
    sentence = "百万円前期比個人消費設備投資" * 2
    token_ids = create_vocab_indexes(vctx.spm, sentence)
    print(f"sentence: {sentence}")
    print(f"token_ids: {token_ids}")
    bow_row = [0] * vctx.size
    for token_id in token_ids:
        if token_id < vctx.size:
            bow_row[token_id] += 1
    print(f"bow_row: {bow_row}")
    assert bow_row[8] == 2
    assert bow_row[22] == 2
    assert bow_row[86] == 2
    assert bow_row[160] == 2

"""
pytest -s -vv src/feature_apis/preprocess/bow_test.py::test_create_tf_idf_record
"""
def test_create_tf_idf_record():
    ctx = ProjectContext()
    row_ids, x, y = load_xy(ctx, VALID_ROWS)
    spm = load_spm_model(ctx)
    document_bow = create_document_bow(x, y, spm)
    document_frequencies = count_document_frequencies(document_bow)
    idf_values = calculate_idf_values(document_frequencies, len(VALID_ROWS.valid_row_ids))
    tf_idf_vectors = [convert_bow_to_tfidf(bow_row, idf_values) for bow_row in document_bow]
    for i, tf_idf_vector in enumerate(tf_idf_vectors):
        print(f"row_id: {VALID_ROWS.valid_row_ids[i]}, tf_idf_vector: {tf_idf_vector}")