
import logging
from pathlib import Path

def setup_logger(outdir="/tmp", log_filename="experiment.log"):
    # 1. ロガーの生成とログレベルの設定
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 既にハンドラが登録されている場合は、二重出力防止のためスキップ
    if logger.hasHandlers():
        return logger

    # 2. 共通のフォーマットを設定（時間、ログレベル、メッセージ）
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3. 標準出力用のハンドラ設定
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # 4. ファイル出力用のハンドラ設定
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)  # ディレクトリがなければ作成
    log_file = out_path / log_filename

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

class ProjectContext:
    def __init__(self):
        self.data_dir = "data"
        self.raw_db_file = f"{self.data_dir}/chABSA/chABSA.db"
        self.pickle_file = f"{self.data_dir}/chABSA/chABSA.pkl"
        self.normalized_db_file = f"{self.data_dir}/chABSA/chABSA_normalized.db"
        self.experiment_outdir = f"{self.data_dir}/experiments"
        self.log_outdir = f"/tmp"
        self.logger = setup_logger(outdir=self.log_outdir, log_filename="experiment.log")

    def info(self, msg):
        self.logger.info(msg)

    def warn(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
