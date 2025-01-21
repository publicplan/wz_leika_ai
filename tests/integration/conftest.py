import time

import pytest
import requests

BASE_URL = "http://0.0.0.0:80"


def get_response(route, base_url=BASE_URL, params=None):
    url = f"{base_url}/{route}"
    if not params:
        params = {}
    r = requests.get(url, params=params)
    r.raise_for_status()

    return r.json()


@pytest.fixture(scope="module")
def api_up():

    attempts = 60
    timeout = 10

    print("\n")
    for n in range(1, attempts + 1):
        try:
            print(f"Trying to connect to API on {BASE_URL} ({n}/{attempts}).")
            r = requests.get(BASE_URL, timeout=1)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout):
            time.sleep(timeout)
            continue
        if r.status_code == 200:
            print(f"API on {BASE_URL} is up.")
            break
    else:
        raise TimeoutError(
            f"API on {BASE_URL} not reachable after {attempts} attempts.")
