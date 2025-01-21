# publicplan

Search engine for WZ (WirtschaftsZweige) and LeiKa (LeistungsKatalog).

Maintainer: Konrad Schultka

# Python environment.

We only support python 3.7. The preferred install methods is using pip. The
package with pinned dependencies can be installed with `make install`. These
dependencies will also be us be used for the CI pipelines and docker builds.
Note that these dependencies contain a pytorch version without CUDA support.

As an alternative, you can create a conda environment (publicplan-env) with
`make conda-env` respectively update it with`make conda-env-update`.

## Getting data

This repo uses [dvc](https://dvc.org/) to synchronize data and model weights,
which can be installed with:

```
pip install dvc
```

The data can then be synced with:

```
dvc pull -r readonly-upstream
```

If you want to add files using dvc, you need to install `dvc[gs]` instead and
use the `upstream`
remote. This requires dida gcloud credentials.

# WZ API

## Docker Container

### CPU

```
make wz
docker run -p 8081:80 wz-api:latest
```

## Code convention

We use pylint and mypy as linters, yapf for code formatting and pytest for unit
tests. The linting and test checks can be run with `make lint` and `make test`
respectively. These checks should pass for each merge request.

You may have to install ```yapf3```, ```python3-yapf``` and ```mypy``` to
execute these commands.

## Add new Datasource

To integrate the new datasource to each request you need to build a dictionary
using the WZ-Code as a key and the appropriate keyword(s) as a list of values.

You can find two implementations under
```publicplan.wz.description.build_gp2019a``` (XML)
and ```publicplan.wz.data_processing.ihk_keywords``` (CSV).

### Add Datasource to Search Results

This previously generate dictionary has to be build in the cli command of the
WZ-API (```publicplan.wz.api.cli```) and must be passed to the
API ```publicplan.wz.api.build_wz_api```. There it can be added to
the ```SearchResult```.

### Providing the Datasource for a training

You just need to build the dictionary in the main method of the ```wz-train```
command (```publicplan.wz.train.cli```). Afterwards you can add the keywords to
the training source.

```
for code, kws in build_gp2019a(keywords_path=WZ_GP2019A_PATH).items():
    descs[code].keywords += kws
```

In a simmilar way you have to extend ```publicplan.wz.train.save_bert_cache```.

## Update train.csv / test.csv / validation.csv

To update the train.csv, test.csv and validation.csv from the main datasource
GWA_Daten_alleWZ_UTF8_20200131.zip you have to execute:

```
python3 -m publicplan.wz.data_processing
```

To compare the newly created files with the old once, you should sort them
beforehand. Otherwise, differences will be obfuscated by to many changes.

## Train

In order to make the training work you need to extract
a ```fasttext_german.zip``` (https://fasttext.cc/docs/en/crawl-vectors.html)
to ```weights/embeddings/fasttext_german```. Afterwards you can
execute ```wz-train --save```. This process takes up to 2-3 days and result is
stored under ```weights/wz/<Ymd-H:M:S>```.

## Used Trained result as default source

Replace the folder ```bert_model``` with the newly created model. Alternatively
you can change the source folder used in the api.py to load the model:

```
checkpoint = WZ_WEIGHTS_DIR.joinpath("<Ymd-H:M:S>_...")
```

## Verify

To run the validation script by executing ```wz-validate```. The script calls
the WZ-API for each entry in the ```data/wz/validation.csv```. As a query Param
the column ```Taetigkeit``` is used. The script excepts that the result contains
the WZ-Code defined in the Column ```FullCode```. The first Column from the
validation file is kept in the result as a reference.

The result is written to a file in the
format ```<Y_m_d_HMS>_validation_result.csv```. The column ```found``` indicates
if the WZ-Code was in the result. A ```0``` means that the WZ-Code was not found
and ```1``` means that the WZ-Code was found. The ```Status``` Column tells if
the request was successful (```OK```) or if an error occurred (```ERROR```).

The ```referenceIndex``` is a link to the original column from the Validation
file.
