from setuptools import find_packages, setup

DESCRIPTION = "AI models for information retrieval "
DESCRIPTION += "from public documents (WZ and LeiKa descriptions)"
setup(name="publicplan-ai",
      author="publicplan GmbH",
      author_email="info@publicplan.de",
      description=DESCRIPTION,
      version="0.1",
      packages=find_packages(include=["publicplan", "publicplan.*"]),
      python_requires="==3.7.*",
      install_requires=[
          "pandas", "numpy", "spacy", "textacy", "symspellpy",
          "torch==1.4.0+cpu", "transformers", "dvc[gs]", "click", "tqdm",
          "pip-tools", "pydantic", "fastapi", "uvicorn",
          "elasticsearch>=6.0.0,<7.0.0", "beautifulsoup4"
      ],
      entry_points={
          "console_scripts": [
              "leika-train=publicplan.leika.train:cli",
              "leika-api=publicplan.leika.api:cli",
              "leika-annotate=publicplan.leika.annotate:cli",
              "wz-train=publicplan.wz.train:cli",
              "wz-validate=publicplan.wz.validate:cli",
              "wz-bert-cache=publicplan.wz.train:save_bert_cache",
              "wz-api=publicplan.wz.api:cli",
          ]
      })
