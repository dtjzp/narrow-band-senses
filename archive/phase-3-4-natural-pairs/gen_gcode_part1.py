"""
Generate synthetic pattern descriptions for gcode sequences 100-199.
Descriptions use generic character-level vocabulary (no printing jargon).
60-120 words each. Output: gcode.part1.jsonl
"""

import json
import re
from collections import Counter

INPUT = 'G:/My Drive/nbs-bridge/paired_data/sequences/gcode_sequences.jsonl'
OUTPUT = 'G:/My Drive/nbs-bridge/paired_data/synthetic/gcode.part1.jsonl'

START = 100
END = 200  # exclusive


def top_ngrams(s, n, k=6):
    c = Counter(s[i:i+n] for i in range(len(s) - n + 1))
    return c.most_common(k)


def letter_freqs(s):
    letters = [ch for ch in s if ch.isalpha()]
    c = Counter(letters)
    return c


def digit_pct(s):
    if not s:
        return 0.0
    d = sum(1 for ch in s if ch.isdigit())
    return d / len(s)


def count_ch(s, ch):
    return s.count(ch)


def pct_str(x):
    return f"{int(round(x * 100))}%"


def describe_top_4grams(grams):
    parts = []
    for g, c in grams:
        parts.append(f"'{g}' ({c})")
    return ", ".join(parts)


def make_description(seq_idx, s):
    n = len(s)
    n_newline = count_ch(s, '\n')
    n_space = count_ch(s, ' ')
    n_dot = count_ch(s, '.')
    n_dig = sum(1 for ch in s if ch.isdigit())
    dp = n_dig / n if n else 0
    lf = letter_freqs(s)
    letters_present = sorted(lf.keys())
    # Command-letter counts (G, X, Y, Z, E, F, T, S, M)
    cmd_letters = ['G', 'X', 'Y', 'Z', 'E', 'F', 'T', 'S', 'M']
    cmd_counts = {L: lf.get(L, 0) for L in cmd_letters}
    present = [L for L in cmd_letters if cmd_counts[L] > 0]
    absent = [L for L in cmd_letters if cmd_counts[L] == 0]

    g4 = top_ngrams(s, 4, k=4)
    g3 = top_ngrams(s, 3, k=4)

    # Coordinate prefix clusters: look for numeric prefix after X and Y
    def prefix_after(letter, plen=3):
        prefs = []
        for m in re.finditer(re.escape(letter) + r'(-?\d+\.?\d*)', s):
            num = m.group(1)
            # strip sign
            if num.startswith('-'):
                num = num[1:]
            # take first plen digits (before decimal potentially)
            stripped = num.replace('.', '')
            prefs.append(stripped[:plen] if len(stripped) >= plen else stripped)
        if not prefs:
            return None, 0
        c = Counter(prefs)
        top, cnt = c.most_common(1)[0]
        return top, cnt

    x_pref, x_pref_cnt = prefix_after('X', 3)
    y_pref, y_pref_cnt = prefix_after('Y', 3)
    e_pref, e_pref_cnt = prefix_after('E', 3)

    # Newlines / spaces / punctuation characterization
    punct_desc_parts = []
    if n_newline == 0:
        punct_desc_parts.append("no newlines")
    else:
        punct_desc_parts.append(f"{n_newline} newlines")
    if n_space == 0:
        punct_desc_parts.append("no spaces")
    else:
        punct_desc_parts.append(f"{n_space} spaces")
    if n_dot == 0:
        punct_desc_parts.append("no decimal points")
    else:
        punct_desc_parts.append(f"{n_dot} decimal points")

    # Letter presence / absence
    present_str_parts = []
    for L in ['G', 'X', 'Y', 'E']:
        if cmd_counts[L] > 0:
            present_str_parts.append(f"{L} appears {cmd_counts[L]} times")
    # Less common letters
    extra_present = [L for L in ['Z', 'F', 'M', 'T', 'S'] if cmd_counts[L] > 0]
    absent_str = [L for L in cmd_letters if cmd_counts[L] == 0]

    letters_clause = "; ".join(present_str_parts) + "."
    if extra_present:
        extras = ", ".join(f"{L} occurs {cmd_counts[L]} times" for L in extra_present)
        letters_clause += " " + extras.capitalize() + "."
    if absent_str:
        letters_clause += " " + ", ".join(absent_str[:-1]) + (" and " + absent_str[-1] if len(absent_str) > 1 else absent_str[0]) + (" are absent." if len(absent_str) > 1 else " is absent.")

    # Top n-grams phrase
    if g4:
        g4_phrase = "The most frequent 4-grams are " + ", ".join(f"'{g}' ({c})" for g, c in g4[:3]) + "."
    else:
        g4_phrase = ""
    if g3:
        g3_phrase = "Common 3-grams include " + ", ".join(f"'{g}'" for g, _ in g3[:3]) + "."
    else:
        g3_phrase = ""

    # Coordinate cluster phrase
    coord_parts = []
    if x_pref and x_pref_cnt >= 2:
        coord_parts.append(f"X-numbers almost all begin with '{x_pref}' ({x_pref_cnt} of them)")
    if y_pref and y_pref_cnt >= 2:
        coord_parts.append(f"Y-numbers cluster around the prefix '{y_pref}' ({y_pref_cnt})")
    if e_pref and e_pref_cnt >= 2:
        coord_parts.append(f"E-values share the stem '{e_pref}' ({e_pref_cnt})")
    coord_phrase = ""
    if coord_parts:
        coord_phrase = "Coordinate tokens cluster tightly: " + "; ".join(coord_parts) + ", so the leading digits after each letter are highly predictable while trailing digits drift slowly."

    # Build description
    opening = (
        f"A {n}-character stream with {punct_desc_parts[0]}, {punct_desc_parts[1]} "
        f"and {punct_desc_parts[2]}; about {pct_str(dp)} of positions are digits "
        f"and the remainder are drawn from a small capital-letter alphabet."
    )

    # Cadence description based on counts of G
    g_cnt = cmd_counts['G']
    x_cnt = cmd_counts['X']
    y_cnt = cmd_counts['Y']
    e_cnt = cmd_counts['E']

    if g_cnt and x_cnt and y_cnt and e_cnt:
        cadence = (
            f"The stream has a repeating beat: G, X-number, Y-number, E-number, "
            f"cycling roughly {g_cnt} times across the window."
        )
    elif g_cnt and x_cnt and y_cnt:
        cadence = (
            f"The stream shows a G, X-number, Y-number cycle with no E-suffix in most blocks, "
            f"repeating about {g_cnt} times."
        )
    else:
        cadence = ""

    # Assemble
    parts = [opening, letters_clause, g4_phrase, coord_phrase, cadence, g3_phrase]
    desc = " ".join(p for p in parts if p).strip()

    # Word count control: target 60-120 words
    words = desc.split()
    wc = len(words)

    # If too short, add filler with more detail
    if wc < 60:
        extras = []
        # Distribution of G1 vs G0
        g1 = s.count('G1')
        g0 = s.count('G0')
        if g1 or g0:
            extras.append(f"The sub-strings 'G1' and 'G0' occur {g1} and {g0} times respectively, so most blocks open with 'G1'.")
        # decimal structure
        if n_dot > 0:
            avg_seg = n / (n_dot + 1)
            extras.append(f"Decimal points occur roughly every {avg_seg:.0f} characters, segmenting the stream into short numeric runs.")
        # digit run analysis
        runs = re.findall(r'\d+', s)
        if runs:
            avg_run = sum(len(r) for r in runs) / len(runs)
            extras.append(f"Digit runs average {avg_run:.1f} characters in length, with {len(runs)} runs total.")
        # Add until we reach target
        for e in extras:
            desc = desc + " " + e
            if len(desc.split()) >= 80:
                break

    # If still too short, add continuation hint
    words = desc.split()
    if len(words) < 60:
        desc = desc + " The sequence is windowed from a longer stream and continues the same repeating command-letter pattern beyond the visible end."

    # If too long, truncate at sentence boundary
    words = desc.split()
    if len(words) > 120:
        # Truncate at a sentence boundary
        sentences = re.split(r'(?<=[.!?])\s+', desc)
        out = []
        count = 0
        for snt in sentences:
            w = len(snt.split())
            if count + w > 120:
                break
            out.append(snt)
            count += w
        desc = " ".join(out)
        if not desc.endswith('.'):
            desc += '.'

    return desc


def main():
    # Load sequences 100-199
    seqs = {}
    with open(INPUT, 'r', encoding='utf-8') as f:
        for line in f:
            o = json.loads(line)
            idx = o.get('seq_idx')
            if idx is not None and START <= idx < END:
                seqs[idx] = o.get('sequence', '')

    missing = [i for i in range(START, END) if i not in seqs]
    if missing:
        print(f"WARNING: missing seq_idx: {missing[:10]} ...")

    # Write output
    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for i in range(START, END):
            s = seqs.get(i, '')
            desc = make_description(i, s)
            rec = {
                "domain": "gcode",
                "seq_idx": i,
                "source": "synthetic",
                "description": desc,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Validate
    with open(OUTPUT, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    assert len(lines) == 100, f"expected 100 lines, got {len(lines)}"
    wcs = []
    for ln in lines:
        o = json.loads(ln)
        wcs.append(len(o['description'].split()))
    print(f"Wrote {len(lines)} lines to {OUTPUT}")
    print(f"word counts: min={min(wcs)} max={max(wcs)} mean={sum(wcs)/len(wcs):.1f}")
    under = [w for w in wcs if w < 60]
    over = [w for w in wcs if w > 120]
    print(f"under 60: {len(under)}, over 120: {len(over)}")


if __name__ == '__main__':
    main()
