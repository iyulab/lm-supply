# CMU Pronouncing Dictionary

`cmudict.dict.gz` is the Carnegie Mellon University Pronouncing Dictionary, unmodified and gzip-compressed. It is embedded
in the `LMSupply.Synthesizer` assembly and read by the built-in English phonemizer.

| | |
|---|---|
| Source | <https://github.com/cmusphinx/cmudict>, file `cmudict.dict` |
| Revision | commit `74790861f652b15e4ac49015a90074ad62a27690` (2025-10-24) |
| Size | 3,618,488 bytes, 135,166 lines; 915,076 bytes compressed |
| SHA-256 of `cmudict.dict` | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` |
| License | 2-clause BSD-style, Copyright (C) 1993-2015 Carnegie Mellon University — see [`LICENSE`](LICENSE), reproduced in the repository's `THIRD-PARTY-NOTICES.md` |

The archive was written with a zero timestamp and no file name, so that the same input gives the same bytes:

```python
import gzip
raw = open("cmudict.dict", "rb").read()
with open("cmudict.dict.gz", "wb") as f:
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=f, mtime=0) as g:
        g.write(raw)
```

A test (`EnglishPhonemizerPartsTests.TheShippedDictionary_IsTheCmuFileUnmodified`) checks the SHA-256 of the decompressed content against the value above.
