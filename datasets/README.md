# Datasets

Each file holds one integer-coded value per line, one line per user, no
header. The domain is the set of distinct values.

| Name (CLI) | File | Values | Users | Source | In repo |
|---|---|---|---|---|---|
| `adult` | `adult.csv` | 74 | 45,222 | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), age column | no |
| `bms` | `BMS.csv` | 256 | 500,000 | BMS click-stream benchmark (KDD Cup 2000), item IDs | no |
| `kosarak` | `kosarak.csv` | 128 | 929,669 | [Kosarak click-stream](http://fimi.uantwerpen.be/data/), item IDs | no |
| `gaussian` | `gaussian_100000.csv` | 60 | 100,000 | synthetic, Gaussian | yes |
| `uniform` | `uniform.csv` | 100 | 100,000 | synthetic, uniform over 0–99 | yes |
| `power` | `power.csv` | 100 | 100,000 | synthetic, power law over 0–99 | yes |

The three real datasets are not redistributed here; download them, export
the column described above one value per line, and place the files in this
folder under the names in the table.
