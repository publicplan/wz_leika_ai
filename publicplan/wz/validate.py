import json
import multiprocessing as mp
import sys
import time
import urllib.parse
from datetime import datetime

import click
import pandas as pd
import tqdm
import urllib3

from publicplan.paths import WZ_VAL_PATH

BASE_URL = 'http://localhost'

FOUND_YES = 1
FOUND_NO = 0


def build_query_url(search):
    return BASE_URL + '/suche/?query=' + urllib.parse.quote(search)


def fetch_query(search):
    query_url = build_query_url(search)
    response = urllib3.PoolManager().request('GET', query_url)
    return json.loads(response.data)


def extract_wz_codes_from_result(json_result):
    if 'results' not in json_result:
        return None

    if len(json_result['results']) != 5:
        raise ValueError('Unexpected amount of results')

    wz_codes = []
    for result in json_result['results']:
        if 'occupation' not in result:
            raise ValueError('Missing occupation in result')
        if 'code' not in result['occupation']:
            raise ValueError('Missing code in result')
        wz_codes.append(result['occupation']['code'])
    return wz_codes


def evaluate_result(job_data):
    (task_name, expected_wz_code, reference_id) = job_data

    # query text
    try:
        data = fetch_query(task_name)
    except json.decoder.JSONDecodeError:
        time.sleep(3)
        return [
            expected_wz_code, '', '', '', '', '', FOUND_NO, 'ERROR',
            reference_id, task_name
        ]

    wz_codes = extract_wz_codes_from_result(data)

    # evaluate result
    if wz_codes is None:
        return [
            expected_wz_code, '', '', '', '', '', FOUND_NO, 'OK', reference_id,
            task_name
        ]

    found = FOUND_NO  # use 0 and 1 to make it easier to create formulas in Excel
    if expected_wz_code in wz_codes:
        found = FOUND_YES
    # [queryWz, resultWz1, resultWz2, resultWz3, resultWz4, resultWz5, found[YES|NO], queryText]
    return [
        expected_wz_code, wz_codes[0], wz_codes[1], wz_codes[2], wz_codes[3],
        wz_codes[4], found, 'OK', reference_id, task_name
    ]


@click.command()
def cli() -> None:
    # read CSV and validate existence of required columns
    data = pd.read_csv(WZ_VAL_PATH, delimiter=',')
    if 'Taetigkeit' not in data.columns:
        raise ValueError('Missing column \'Taetigkeit\'')
    if 'FullCode' not in data.columns:
        raise ValueError('Missing column \'FullCode\'')
    if 'Unnamed: 0' not in data.columns:
        raise ValueError('Missing column \'Unnamed: 0\'')

    validation_item_count = len(data['FullCode'])

    # prepare data to be processed parallel
    validation_items = []
    for i in range(0, validation_item_count):
        validation_items.append((data['Taetigkeit'][i], data['FullCode'][i],
                                 data['Unnamed: 0'][i]))

    # query each text from the validation.csv against the API
    pool = mp.Pool(mp.cpu_count())
    results = []
    items = pool.imap_unordered(evaluate_result, validation_items)
    for result in tqdm.tqdm(items, total=validation_item_count):
        results.append(result)

    # generate output file
    filename = datetime.now().strftime(
        "%Y_%m_%d_%H%M%S") + '_validation_result.csv'
    pd.DataFrame(results,
                 columns=['queryWz', 'WZ1', 'WZ2', 'WZ3', 'WZ4', 'WZ5', 'found',
                          'status', 'queryText', 'referenceIndex']) \
        .to_csv(filename)
    sys.stdout.write('Result was written to ' + filename + '\n')
