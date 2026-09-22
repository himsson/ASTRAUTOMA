"""Скачивает чертежи KSP (.craft) с GitHub для школы конструктора.

Ищет файлы .craft поиском кода GitHub (через `gh`, нужен вход в gh),
берёт только чертежи вертикальной сборки (type = VAB — ракеты, спутники,
станции, базы, роверы; самолёты собираются в SPH и сюда не попадают),
и складывает их в data/craft_library/github/<владелец>__<репозиторий>/.

Поиск кода отдаёт не больше 1000 результатов на запрос, поэтому запросы
режутся по размеру файла. Одинаковые файлы (по содержимому) не хранятся
дважды. Остановиться можно на лимите объёма.

    python tools/fetch_crafts.py [--limit-mb 1024]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "data" / "craft_library" / "github"
INDEX = LIB / "index.json"
SIZES = ["<8000", "8000..16000", "16000..30000", "30000..50000", "50000..80000", "80000..120000",
         "120000..200000", "200000..350000", "350000..600000", ">600000"]


def gh(args: list[str]) -> dict:
    for attempt in range(6):
        r = subprocess.run(["gh", "api", "-X", "GET"] + args, capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0:
            return json.loads(r.stdout)
        if "rate limit" in (r.stderr + r.stdout).lower() or "403" in r.stderr:
            time.sleep(35 + 15 * attempt)
            continue
        if "422" in r.stderr:
            return {}
        time.sleep(5)
    return {}


def search(size: str, page: int) -> list[dict]:
    q = f'extension:craft "type = VAB" size:{size}'
    d = gh(["search/code", "-f", f"q={q}", "-f", "per_page=100", "-f", f"page={page}"])
    return d.get("items", []) if d else []


def raw_url(item: dict) -> str:
    repo = item["repository"]["full_name"]
    sha = item["url"].split("ref=")[-1] if "ref=" in item["url"] else "HEAD"
    return f"https://raw.githubusercontent.com/{repo}/{sha}/{urllib.parse.quote(item['path'])}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-mb", type=float, default=1024.0)
    a = ap.parse_args()
    LIB.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else {"files": {}, "hashes": []}
    hashes = set(index["hashes"])
    total = sum(f["bytes"] for f in index["files"].values())
    t0 = time.time()
    got = skipped = 0
    for size in SIZES:
        for page in range(1, 11):
            items = search(size, page)
            if not items:
                break
            for it in items:
                key = f"{it['repository']['full_name']}/{it['path']}"
                if key in index["files"]:
                    continue
                try:
                    with urllib.request.urlopen(raw_url(it), timeout=30) as resp:
                        data = resp.read()
                except Exception:
                    skipped += 1
                    continue
                h = hashlib.sha1(data).hexdigest()
                text = data[:4000].decode("utf-8", "ignore")
                if h in hashes or not re.search(r"^\s*type\s*=\s*VAB", text, re.M):
                    skipped += 1
                    continue
                folder = LIB / it["repository"]["full_name"].replace("/", "__")
                folder.mkdir(parents=True, exist_ok=True)
                name = re.sub(r'[<>:"/\\|?*]', "_", Path(it["path"]).name)
                dest = folder / name
                n = 1
                while dest.exists():
                    dest = folder / f"{Path(name).stem}_{n}.craft"
                    n += 1
                dest.write_bytes(data)
                hashes.add(h)
                index["files"][key] = {"file": str(dest.relative_to(LIB)), "bytes": len(data), "sha1": h}
                total += len(data)
                got += 1
                if got % 25 == 0:
                    index["hashes"] = sorted(hashes)
                    INDEX.write_text(json.dumps(index, indent=1), encoding="utf-8")
                    print(f"  скачано {got}, всего {len(index['files'])} файлов, {total / 1e6:.1f} МБ "
                          f"[{time.time() - t0:.0f} с]", flush=True)
                if total >= a.limit_mb * 1e6:
                    break
            time.sleep(7)                      # поиск кода: не больше 10 запросов в минуту
            if total >= a.limit_mb * 1e6:
                break
        print(f"размер {size}: всего {len(index['files'])} файлов, {total / 1e6:.1f} МБ", flush=True)
        if total >= a.limit_mb * 1e6:
            break
    index["hashes"] = sorted(hashes)
    INDEX.write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"готово: +{got} новых, пропущено {skipped}, всего {len(index['files'])} файлов, {total / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
