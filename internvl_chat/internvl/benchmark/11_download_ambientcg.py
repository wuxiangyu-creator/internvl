"""
Download ~150 materials from AmbientCG (1K resolution PBR materials).

For each material, we keep only the Color/Albedo map as our "texture",
since that matches the format of our in-domain texture.png files.
"""

import argparse
import json
import os
import requests
import time
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile


API_URL = "https://ambientcg.com/api/v2/full_json"


def fetch_material_list(n_materials: int, sort: str = "Popular") -> list:
    """Fetch list of materials from AmbientCG API."""
    materials = []
    offset = 0
    page_size = 100
    while len(materials) < n_materials:
        params = {
            "type": "Material",
            "include": "downloadData",
            "limit": page_size,
            "offset": offset,
            "sort": sort,
        }
        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        found = data.get("foundAssets", [])
        if not found:
            break
        materials.extend(found)
        offset += page_size
        time.sleep(0.5)
    return materials[:n_materials]


def download_color_map(material: dict, out_dir: Path) -> str:
    """
    Download the Color/Albedo map for a material.
    Returns the local path to the saved image, or None on failure.
    """
    asset_id = material["assetId"]
    downloads = material.get("downloadFolders", {}).get("default", {}).get("downloadFiletypeCategories", {}).get("zip", {}).get("downloads", [])

    # Pick the smallest 1K resolution zip
    candidate = None
    for d in downloads:
        attrs = d.get("attribute", "")
        if "1K-JPG" in attrs or "1K-PNG" in attrs:
            candidate = d
            break
    if candidate is None:
        return None

    zip_url = candidate.get("downloadLink") or candidate.get("fullDownloadPath")
    if zip_url is None:
        return None

    zip_path = out_dir / f"{asset_id}.zip"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(zip_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    f.write(chunk)
    except Exception as e:
        print(f"  [SKIP] {asset_id}: download failed: {e}")
        return None

    # Extract Color/Albedo map
    color_path = out_dir / f"{asset_id}_color.png"
    try:
        with ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            color_candidates = [n for n in names
                                if any(k in n.lower() for k in ("_color.", "_albedo.", "_diff."))]
            if not color_candidates:
                # Fallback: any image file
                color_candidates = [n for n in names
                                    if n.lower().endswith((".jpg", ".png"))]
            if not color_candidates:
                print(f"  [SKIP] {asset_id}: no color map in zip")
                return None
            with zf.open(color_candidates[0]) as f_in:
                content = f_in.read()
            with open(color_path, "wb") as f_out:
                f_out.write(content)
    except Exception as e:
        print(f"  [SKIP] {asset_id}: extract failed: {e}")
        return None
    finally:
        if zip_path.exists():
            zip_path.unlink()

    return str(color_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_materials", type=int, default=150)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {args.n_materials} materials from AmbientCG...")
    materials = fetch_material_list(args.n_materials)
    print(f"Got {len(materials)} materials.")

    manifest = []
    failed = 0
    for i, m in enumerate(materials):
        aid = m["assetId"]
        print(f"  [{i+1}/{len(materials)}] downloading {aid}...")
        local_path = download_color_map(m, out_dir)
        if local_path is None:
            failed += 1
            continue
        manifest.append({
            "asset_id": aid,
            "category": m.get("category") or "unknown",
            "texture_path": os.path.relpath(local_path, args.out_dir),
        })
        time.sleep(0.3)  # be polite

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDownloaded {len(manifest)} materials (failed {failed})")
    print(f"  Output: {out_dir}")
    print(f"  Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
