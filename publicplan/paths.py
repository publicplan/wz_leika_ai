from datetime import datetime
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parents[1].joinpath("weights")
EMBEDDINGS_DIR = WEIGHTS_DIR.joinpath("embeddings")

IWNLP_PATH = Path(__file__).parents[1].joinpath(
    "data", "nlp", "IWNLP.Lemmatizer_20181001.json")

BASE_SPELLDICT_PATH = Path(__file__).parents[1].joinpath(
    "data", "nlp", "internet-de-forms.num")
SPELLDICT_PATH = Path(__file__).parents[1].joinpath("data", "nlp",
                                                    "word_freq.txt")

LEIKA_DATA_DIR = Path(__file__).parents[1].joinpath("data", "leika")
LEIKA_DESCS_PATH = LEIKA_DATA_DIR.joinpath("leika.json")
LEIKA_CLEANED_DESCS_PATH = LEIKA_DATA_DIR.joinpath("leika_cleaned.json")

LEIKA_DATA_PATH = LEIKA_DATA_DIR.joinpath("data.csv")
LEIKA_TRAIN_PATH = LEIKA_DATA_DIR.joinpath("train.json")
LEIKA_VAL_PATH = LEIKA_DATA_DIR.joinpath("validation.json")
LEIKA_TEST_PATH = LEIKA_DATA_DIR.joinpath("test.json")
LEIKA_RES_PATH = LEIKA_DATA_DIR.joinpath("reserved.json")

LEIKA_WEIGHTS_DIR = Path(__file__).parents[1].joinpath("weights", "leika")

WZ_DATA_DIR = Path(__file__).parents[1].joinpath("data", "wz")
WZ_DESCS_PATH = WZ_DATA_DIR.joinpath(
    "WZ2008-2019-10-22-Structure_with_explanatory_notes.xml")
WZ_KEYWORDS_PATH = WZ_DATA_DIR.joinpath("WZ2008-2019-10-22-Keywords.xml")
WZ_GP2019A_PATH = WZ_DATA_DIR.joinpath("WZ2008-2021-08-02-Correspondences.xml")
WZ_DATA_ZIP = WZ_DATA_DIR.joinpath("")
WZ_DATA_PATH = WZ_DATA_DIR.joinpath("")
WZ_IHK_PATH = WZ_DATA_DIR.joinpath("")

WZ_TRAIN_PATH = WZ_DATA_DIR.joinpath("train.csv")
WZ_VAL_PATH = WZ_DATA_DIR.joinpath("validation.csv")
WZ_TEST_PATH = WZ_DATA_DIR.joinpath("test.csv")

WZ_WEIGHTS_DIR = Path(__file__).parents[1].joinpath("weights", "wz")

MOCK_DATA_DIR = Path(__file__).parents[1].joinpath("tests", "mock_data")


def checkpoint_dir(base_dir: Path) -> Path:
    """Create and return timestamped directory under base_dir."""

    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    checkpoint = base_dir.joinpath(timestamp)
    checkpoint.mkdir(parents=True, exist_ok=True)

    return checkpoint
