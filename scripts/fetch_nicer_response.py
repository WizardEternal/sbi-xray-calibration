"""Fetch the public NICER XTI on-axis response into data/nicer/ so the
`response: NICER` configs can run. The XMM EPIC-pn response ships with jaxspec;
NICER's does not, so we pull the standard on-axis ARF + RMF from the public
NICER CALDB (a few MB total).
"""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://heasarc.gsfc.nasa.gov/FTP/caldb/data/nicer/xti/cpf"
FILES = {
    "nicer.arf": f"{BASE}/arf/nixtiaveonaxis20170601v005.arf",
    "nicer.rmf": f"{BASE}/rmf/nixtiref20170601v003.rmf",
}


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "nicer"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = out / name
        if dst.exists():
            print(f"[skip] {dst} exists")
            continue
        print(f"[fetch] {url} -> {dst}")
        urlretrieve(url, dst)
    print("done. NICER response in", out)


if __name__ == "__main__":
    main()
