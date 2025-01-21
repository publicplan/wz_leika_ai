import re
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup

from publicplan.paths import WZ_DATA_DIR, WZ_KEYWORDS_PATH

DESTATIS_URL = "https://www.klassifikationsserver.de/klassService/jsp/common/url.jsf?variant=wz2008"
#pylint: disable=line-too-long
KEYWORDS_FILE_URL = "https://www.klassifikationsserver.de/klassService/jsp/variant/downloadexport?type=EXPORT_XML_SEARCH_WORDS&variant=wz2008&language=DE"


def destatis_keywords_date() -> datetime:
    """Scrape date of keyword update from destatis site."""
    response = requests.get(DESTATIS_URL)
    if not response.status_code == 200:
        raise ConnectionError("Error in checking keywords date.")
    soup = BeautifulSoup(response.content, "html.parser")
    search_re = re.compile("Letzte Aktualisierung der Stichwörter.*")
    result = soup.find(text=search_re)
    if result is None or result.next is None:
        raise ValueError("Can not find keyword update date.")
    try:
        date_str = result.next.text.split(" ")[0]
        date = datetime.strptime(date_str, "%d.%m.%Y")
    except ValueError:
        raise ValueError("Error in parsing keyword update date.")
    return date


def saved_keywords_date() -> datetime:
    """Date of saved keywords list"""
    file_name = WZ_KEYWORDS_PATH.name
    date_str = "-".join(file_name.split("-")[1:4])
    date = datetime.strptime(date_str, "%Y-%m-%d")

    return date


def download_keyword_list() -> Path:
    """Download new keyword list from destatis.

    Returns:
        Path to new keywords xml file.
    """
    response = requests.get(KEYWORDS_FILE_URL)
    zip_path = WZ_DATA_DIR.joinpath("keywords.zip")
    with open(zip_path, "wb") as f:
        f.write(response.content)

    zf = ZipFile(zip_path)
    keywords_files = [name for name in zf.namelist() if "Keywords" in name]
    if not keywords_files:
        raise ValueError("Error in extracting updated keywords list.")
    keywords_file = keywords_files[0]
    zf.extract(keywords_file, path=str(WZ_DATA_DIR))
    keywords_path = WZ_DATA_DIR.joinpath(keywords_file)
    if not keywords_path.exists():
        raise ValueError("Error in extracting updated keywords list.")

    return keywords_path
