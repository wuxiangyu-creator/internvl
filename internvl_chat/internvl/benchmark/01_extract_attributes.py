"""
Extract structured attributes from material descriptions.

Strategy:
  1. Rule-based keyword matching (fast, covers ~90% with high precision)
  2. Fallback to a tiny LLM call for ambiguous cases (optional)

Output adds an "attributes" field to each JSONL record:
  {
    "id": "...",
    "image": "...",
    "texture": "...",
    "description": "...",
    "attributes": {
        "category": "wood" | "metal" | "fabric" | ...,
        "color":    "brown" | "grey" | ...,
        "finish":   "matte" | "glossy" | "metallic" | ...,
        "roughness":"smooth" | "medium" | "coarse",
        "pattern":  "solid" | "striped" | "mottled" | "fibrous" | ...,
        "_source":  "rule" | "llm" | "unknown"
    }
  }
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ── Category keywords (ordered by priority) ──────────────────────────────────
CATEGORY_KEYWORDS = {
    "metal":   ["metal", "metallic", "steel", "iron", "brass", "copper", "gold", "silver",
                "aluminum", "chrome", "rust", "alloy"],
    "wood":    ["wood", "wooden", "timber", "bark", "plank", "grain", "oak", "pine",
                "mahogany", "birch"],
    "stone":   ["stone", "rock", "marble", "granite", "concrete", "pebble", "gravel",
                "boulder", "slate", "cobble"],
    "fabric":  ["fabric", "cloth", "textile", "cotton", "linen", "wool", "silk", "felt",
                "weave", "woven", "knit"],
    "leather": ["leather", "hide", "suede"],
    "plastic": ["plastic", "rubber", "vinyl", "synthetic polymer"],
    "glass":   ["glass", "transparent", "translucent", "crystal"],
    "ceramic": ["ceramic", "porcelain", "tile", "pottery", "clay"],
    "paper":   ["paper", "cardboard", "parchment"],
    "skin":    ["skin", "flesh", "scale", "feather", "fur"],
    "ground":  ["soil", "dirt", "mud", "sand", "earth", "ground"],
    "plant":   ["leaf", "leaves", "grass", "moss", "vegetation", "foliage"],
}

# ── Color keywords ───────────────────────────────────────────────────────────
COLOR_KEYWORDS = {
    "brown":  ["brown", "tan", "beige", "khaki", "chestnut", "umber"],
    "grey":   ["grey", "gray", "silver", "ash", "charcoal"],
    "black":  ["black", "ebony", "obsidian"],
    "white":  ["white", "ivory", "cream", "off-white", "pale"],
    "red":    ["red", "crimson", "scarlet", "maroon", "burgundy"],
    "orange": ["orange", "amber", "rust", "copper-color"],
    "yellow": ["yellow", "gold", "golden", "ochre"],
    "green":  ["green", "olive", "emerald", "jade"],
    "blue":   ["blue", "navy", "teal", "azure", "cobalt"],
    "purple": ["purple", "violet", "lavender", "magenta"],
    "pink":   ["pink", "rose", "salmon"],
}

# ── Finish keywords ──────────────────────────────────────────────────────────
FINISH_KEYWORDS = {
    "glossy":    ["glossy", "shiny", "polished", "reflective", "lustrous", "high-gloss",
                  "mirror-like"],
    "metallic":  ["metallic"],
    "matte":     ["matte", "matt", "flat", "non-reflective", "dull"],
    "satin":     ["satin", "semi-gloss", "silky", "soft sheen"],
    "transparent": ["transparent", "translucent", "see-through"],
}

# ── Roughness keywords ───────────────────────────────────────────────────────
ROUGHNESS_KEYWORDS = {
    "coarse": ["coarse", "rough", "rugged", "gritty", "jagged", "abrasive", "uneven",
               "bumpy"],
    "medium": ["textured", "grainy", "fibrous"],
    "smooth": ["smooth", "polished", "flat", "even", "soft", "fine"],
}

# ── Pattern keywords ─────────────────────────────────────────────────────────
PATTERN_KEYWORDS = {
    "striped":   ["striped", "stripe", "lined", "parallel lines"],
    "checkered": ["checkered", "checker", "grid", "tiled pattern"],
    "mottled":   ["mottled", "speckled", "spotted", "dappled", "patchy"],
    "fibrous":   ["fibrous", "woven", "thread", "fiber"],
    "cracked":   ["crack", "fissure", "broken", "weathered"],
    "scaled":    ["scale", "scaly", "fish-like"],
    "solid":     ["uniform", "solid", "monochrome", "consistent"],
}


def match_attribute(text: str, kw_dict: dict) -> str:
    """Return first matching label or 'unknown'."""
    text_lower = text.lower()
    for label, keywords in kw_dict.items():
        for kw in keywords:
            # word-boundary match for short keywords, substring otherwise
            if len(kw) <= 4:
                if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                    return label
            else:
                if kw in text_lower:
                    return label
    return "unknown"


def extract_attributes(description: str) -> dict:
    """Rule-based attribute extraction from a single description."""
    return {
        "category":  match_attribute(description, CATEGORY_KEYWORDS),
        "color":     match_attribute(description, COLOR_KEYWORDS),
        "finish":    match_attribute(description, FINISH_KEYWORDS),
        "roughness": match_attribute(description, ROUGHNESS_KEYWORDS),
        "pattern":   match_attribute(description, PATTERN_KEYWORDS),
        "_source":   "rule",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+",
                        help="Input JSONL file(s) (will be concatenated)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file with attributes added")
    parser.add_argument("--stats", default=None,
                        help="Optional path to write per-attribute distribution stats")
    args = parser.parse_args()

    # Load samples
    all_samples = []
    for input_path in args.input:
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
    print(f"Loaded {len(all_samples)} samples from {len(args.input)} file(s)")

    # Extract attributes
    counters = {k: Counter() for k in ["category", "color", "finish", "roughness", "pattern"]}
    for sample in all_samples:
        attrs = extract_attributes(sample.get("description", ""))
        sample["attributes"] = attrs
        for k, c in counters.items():
            c[attrs[k]] += 1

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_samples)} samples to {args.output}")

    # Stats summary
    print("\n── Attribute distribution ──────────────────────────────────────")
    for k, c in counters.items():
        total = sum(c.values())
        unknown_pct = 100 * c["unknown"] / total
        top5 = c.most_common(6)
        print(f"\n{k}  (unknown={unknown_pct:.1f}%):")
        for label, n in top5:
            print(f"  {label:>12s}  {n:>6d}  ({100*n/total:5.1f}%)")

    if args.stats:
        with open(args.stats, "w") as f:
            json.dump({k: dict(c) for k, c in counters.items()}, f, indent=2)
        print(f"\nFull stats written to {args.stats}")


if __name__ == "__main__":
    main()
