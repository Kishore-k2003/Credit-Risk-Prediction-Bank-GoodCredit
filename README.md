# Credit Risk Prediction

This project analyzes customer credit data and trains credit-risk prediction
models. The analysis notebook is in `notebooks/`, and reusable processing code
is in `src/`.

## Database configuration

Set these environment variables before running database extraction or the
notebook:

- `DB_HOST`
- `DB_PORT`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`

The generated customer-level dataset is intentionally excluded from version
control because it contains personal and financial information. The notebook
is committed without saved cell outputs.
