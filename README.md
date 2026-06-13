

# 環境構築
```bash
python -m venv /opt/venvs/.venv
source /opt/venvs/.venv/bin/activate
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir chabsa-document-classification
cd chabsa-document-classification
git init

```

# gitパスワード入力を省略
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
# 以降、git push などでパスワード入力が不要になる
```

# 実行順序
## chABSA ダウンロード

```bash
# プロジェクトディレクトリから実行
python src/feature_apis/dataset/download_chABSA.py download
python src/feature_apis/dataset/download_chABSA.py enum_keys
python src/feature_apis/dataset/download_chABSA.py split_save
python src/feature_apis/dataset/download_chABSA.py sample

# download して enum_keys でキー等を確認
# split_save で SQLite DB に保存（data/chABSA/chABSA.db）
# sample でサンプルデータを閲覧
```

## 正規化

```bash
# プロジェクトディレクトリから実行
python src/feature_apis/normalize/normalize.py normalize
python src/feature_apis/normalize/normalize.py view_samples

# normalize で正規化処理を実行、SQLiteに保存される（data/chABSA/chABSA_normalized.db）
# view_samples で正規化前後のサンプルを閲覧
```

## トークナイズとベンチマーク

```bash
# プロジェクトディレクトリから実行
python src/feature_apis/tokenize/benchmark_vocab_size/benchmark_vocab_size.py

# data/benchmark_vocab_size にモデルとトークナイザのベンチマーク結果が保存される
# ベンチマークの結果を参考に、tokenize.py の vocab_size を適切な値に設定する
python src/feature_apis/tokenize/tokenize.py 2959
```

## まとめ
```bash
python src/feature_apis/dataset/download_chABSA.py download
python src/feature_apis/dataset/download_chABSA.py split_save
python src/feature_apis/normalize/normalize.py normalize
python src/feature_apis/tokenize/sp_tokenize.py 2959
```

# ここまで前処理


# BoW
# BoW + TF-IDF
# Word2Vec 平均
# Word2Vec TF-IDF 加重平均
# Word2Vec クラスタ分布
# 【独自】文脈ベクトルクラスタ分布（位置重み無し）
# 【独自】文脈ベクトルクラスタ分布（位置重み有り）

