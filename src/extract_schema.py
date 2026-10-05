import os

import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": int(os.environ["DB_PORT"]),
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "database": os.environ["DB_NAME"],
}


def build_connection_uri(cfg: dict) -> str:
    user = quote_plus(cfg["user"])
    password = quote_plus(cfg["password"])
    return (
        f"mysql+pymysql://{user}:{password}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )


def main() -> None:
    engine = create_engine(build_connection_uri(DB_CONFIG), pool_pre_ping=True)

    with engine.begin() as conn:
        tables = pd.read_sql(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema_name
                ORDER BY table_name
                """
            ),
            conn,
            params={"schema_name": DB_CONFIG["database"]},
        )

        columns = pd.read_sql(
            text(
                """
                SELECT
                    table_name,
                    column_name,
                    data_type,
                    is_nullable,
                    column_type,
                    ordinal_position
                FROM information_schema.columns
                WHERE table_schema = :schema_name
                ORDER BY table_name, ordinal_position
                """
            ),
            conn,
            params={"schema_name": DB_CONFIG["database"]},
        )

        row_counts = []
        for table_name in tables["table_name"].tolist():
            cnt = conn.execute(text(f"SELECT COUNT(*) AS cnt FROM {table_name}"))
            row_counts.append({"table_name": table_name, "row_count": cnt.scalar_one()})

        row_counts = pd.DataFrame(row_counts)

    schema_overview = columns.merge(row_counts, on="table_name", how="left")

    tables.to_csv("outputs/tables.csv", index=False)
    columns.to_csv("outputs/columns.csv", index=False)
    row_counts.to_csv("outputs/row_counts.csv", index=False)
    schema_overview.to_csv("outputs/schema_overview.csv", index=False)

    print("Connected successfully")
    print("Tables found:")
    print(tables.to_string(index=False))
    print("\nRow counts:")
    print(row_counts.to_string(index=False))


if __name__ == "__main__":
    main()
