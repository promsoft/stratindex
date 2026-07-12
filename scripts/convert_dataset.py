"""One-off conversion of the upstream cpsmarch2015 dataset to csv.gz.

Downloads ``data/cpsmarch2015.rda`` from the R package repository
(https://github.com/xiangzhou09/strat) and writes
``src/stratindex/data/cpsmarch2015.csv.gz``, preserving factor labels.

Usage: python scripts/convert_dataset.py [path/to/cpsmarch2015.rda]
"""

import gzip
import sys
import tempfile
import urllib.request
from pathlib import Path

import pyreadr

UPSTREAM = "https://raw.githubusercontent.com/xiangzhou09/strat/master/data/cpsmarch2015.rda"
OUT = Path(__file__).resolve().parent.parent / "src" / "stratindex" / "data" / "cpsmarch2015.csv.gz"


def main() -> None:
    if len(sys.argv) > 1:
        rda_path = Path(sys.argv[1])
    else:
        rda_path = Path(tempfile.gettempdir()) / "cpsmarch2015.rda"
        urllib.request.urlretrieve(UPSTREAM, rda_path)

    df = pyreadr.read_r(str(rda_path))["cpsmarch2015"]
    assert list(df.columns) == ["income", "big_class", "micro_class", "education", "weight"]
    assert len(df) == 14358
    if (df["micro_class"] % 1 == 0).all():
        df["micro_class"] = df["micro_class"].astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the gzip output reproducible
    with gzip.GzipFile(OUT, "wb", mtime=0) as fh:
        df.to_csv(fh, index=False)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(df)} rows)")


if __name__ == "__main__":
    main()
