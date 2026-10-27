"""Run the preparation of one source and write its CSV."""
import argparse

from data.djinni import DjinniSource
from data.linkedin import LinkedInSource

SOURCES = {"djinni": DjinniSource, "linkedin": LinkedInSource}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=SOURCES)
    args = parser.parse_args()

    source = SOURCES[args.source]()
    df = source.prepare()
    source.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(source.out, index=False, encoding="utf-8-sig")
    print(f"{len(df)} rows -> {source.out}")
    role_col = "weak_role" if "weak_role" in df else "role"
    print(df[role_col].value_counts())


if __name__ == "__main__":
    main()