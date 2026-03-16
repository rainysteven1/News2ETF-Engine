"""Convert CSV and XLSX files in data/ to Parquet format in data/converted/."""

from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "converted"


def convert_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(DATA_DIR.glob("*.csv"))
    xlsx_files = list(DATA_DIR.glob("*.xlsx"))

    for f in csv_files:
        print(f"Converting CSV: {f.name}")
        df = pl.read_csv(f)
        out = OUTPUT_DIR / f"{f.stem}.parquet"
        df.write_parquet(out)
        print(f"  -> {out.name}  ({len(df)} rows)")

    for f in xlsx_files:
        import fastexcel

        wb = fastexcel.read_excel(f)
        sheet_names = wb.sheet_names
        print(f"Converting XLSX: {f.name}  ({len(sheet_names)} sheets)")
        for sheet_name in sheet_names:
            df = pl.read_excel(f, sheet_name=sheet_name)
            safe_name = sheet_name.replace("/", "_").replace("\\", "_")
            out = OUTPUT_DIR / f"{f.stem}_{safe_name}.parquet"
            df.write_parquet(out)
            print(f"  -> {out.name}  ({len(df)} rows)")

    print(f"\nDone. {len(csv_files) + len(xlsx_files)} files converted.")


if __name__ == "__main__":
    convert_all()
