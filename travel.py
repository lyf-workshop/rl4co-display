"""
travel.py
-----------
小工具：利用 OpenStreetMap 的 Nominatim 接口批量查询景区或地标的坐标。

支持两种输入方式：
1. 命令行参数（`--places "黄山|黄山市|China" "西湖|杭州|China"`）
2. 文本文件（每行 `名称 | 城市 | 国家`，`|` 两边可以有空格，空白行与 `# 注释` 会跳过）
3. 组合模式：可以同时给定命令行 + 文件。

每次请求会自动等待一段时间（默认1秒），以避免触发 Nominatim 的速率限制。结果可打印在终端，也可以输出到 JSON 文件。

注意：Nominatim 要求请求附带 User-Agent，复制、部署或批量爬取时请遵守其使用政策。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "RL4CO-TravelCoordinates/1.0 (https://github.com/org/rl4co-display)"


@dataclass
class PlaceEntry:
    """代表一个待查询的景区或地标。"""

    name: str
    city: Optional[str] = None
    country: Optional[str] = None

    @property
    def query(self) -> str:
        """拼接最终查询字符串（逗号分隔）。"""
        tokens = [self.name]
        if self.city:
            tokens.append(self.city)
        if self.country:
            tokens.append(self.country)
        return ", ".join(tokens)


def fetch_coordinates(
    query: str,
    session: requests.Session,
    country_code: Optional[str] = None,
    retries: int = 3,
) -> Optional[dict]:
    """
    通过 Nominatim 查询地理坐标，返回与查询最匹配的第一条结果。
    """

    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
    }
    if country_code:
        params["countrycodes"] = country_code

    for attempt in range(retries):
        response = session.get(NOMINATIM_URL, params=params, timeout=20)
        if response.status_code == 200:
            payload = response.json()
            if payload:
                return payload[0]
            return None
        if response.status_code in {429, 503}:
            wait = 1 + attempt * 2
            print(f"[!] {response.status_code} 速率限制，等待 {wait}s 后重试……")
            time.sleep(wait)
            continue
        response.raise_for_status()

    return None


def load_from_file(path: Path) -> List[PlaceEntry]:
    """根据固定格式读取文件，默认以 `|` 分隔 name/city/country。"""

    entries: List[PlaceEntry] = []
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在")

    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split("|")]
            name = parts[0] if parts else ""
            city = parts[1] if len(parts) >= 2 and parts[1] else None
            country = parts[2] if len(parts) >= 3 and parts[2] else None
            if not name:
                continue
            entries.append(PlaceEntry(name=name, city=city, country=country))
    return entries


def parse_cli_place(raw: str) -> Optional[PlaceEntry]:
    """命令行参数也使用 `|` 分隔，避免名称中自带逗号的歧义。"""

    if not raw:
        return None
    parts = [part.strip() for part in raw.split("|")]
    name = parts[0]
    if not name:
        return None
    city = parts[1] if len(parts) >= 2 and parts[1] else None
    country = parts[2] if len(parts) >= 3 and parts[2] else None
    return PlaceEntry(name=name, city=city, country=country)


def query_places(
    entries: Iterable[PlaceEntry],
    pause: float,
    country_code: Optional[str],
) -> List[dict]:
    """轮询每个景区并保存查询结果。"""

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "zh-CN,zh;q=0.9, en;q=0.8",
        }
    )

    results: List[dict] = []
    for entry in entries:
        query_text = entry.query
        print(f"查询：{query_text}")
        data = fetch_coordinates(query_text, session=session, country_code=country_code)
        if data:
            lat = float(data["lat"])
            lon = float(data["lon"])
            result = {
                "name": entry.name,
                "city": entry.city,
                "country": entry.country,
                "query": query_text,
                "display_name": data.get("display_name"),
                "lat": lat,
                "lon": lon,
                "boundingbox": data.get("boundingbox"),
            }
            print(f"  → 坐标：{lat:.6f}, {lon:.6f}")
            results.append(result)
        else:
            print(f"  → 未找到结果，请检查名称或尝试扩大描述。")
        time.sleep(max(0, pause))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="使用 Nominatim 查询景区/地标坐标并输出 JSON。",
        epilog="示例：python travel.py -p \"黄山|黄山|China\" -p \"西湖|杭州|China\"",
    )
    parser.add_argument(
        "-p",
        "--places",
        nargs="+",
        help="命令行输入的景区名称，格式：名称|城市|国家（国家可选）。",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="包含景区信息的文本文件，每行 NAME | CITY | COUNTRY（| 可缺省）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="可选，写入查询结果的 JSON 文件（覆盖）。",
    )
    parser.add_argument(
        "-c",
        "--country-code",
        help="可选，ISO 3166-1 alpha-2 国家码，用于限制搜索区域（如 CN）。",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="每次请求之间的等待时间（秒），默认 1s，避免触发速率限制。",
    )

    args = parser.parse_args()

    entries: List[PlaceEntry] = []

    if args.places:
        for raw in args.places:
            entry = parse_cli_place(raw)
            if entry:
                entries.append(entry)

    if args.file:
        entries.extend(load_from_file(args.file))

    if not entries:
        parser.error("请通过 --places 或 --file 提供至少一个景区。")

    results = query_places(entries=entries, pause=args.pause, country_code=args.country_code)

    if args.output:
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(results, stream, ensure_ascii=False, indent=2)
        print(f"已将 {len(results)} 条结果写入 {args.output}")
    else:
        print("\n已完成查询。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断执行。")
        sys.exit(1)