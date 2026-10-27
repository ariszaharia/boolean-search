"""
Greek letters -> their names (decision D7 in targeted_set_design.md, 2026-10-27).

The engine's tokenizer splits words at every non-ASCII byte, so "TNF-α" used to be indexed as "tnf" and the
α was lost. Every Greek letter is now written as its name, once, when the corpus and the query files are
prepared, so every system (B1, B2, B3, E) sees the same text.

A Greek letter is its own word: the name gets a space on any side where it touches a letter or digit.
    "TNF-α" -> "TNF-alpha"     "α-synuclein" -> "alpha-synuclein"     "IFNγ" -> "IFN gamma"
    "TGFβ1" -> "TGF beta 1"    "10 μM" -> "10 mu M"
This matches how the engine treats a hyphen ("TNF-alpha" and "TNF alpha" give the same words).
The micro sign µ (U+00B5, a different character that looks the same as μ) is treated as mu.

Run it to check the examples:   python api/greek.py
"""
NAMES = {
    'alpha': 'αΑ', 'beta': 'βΒ', 'gamma': 'γΓ', 'delta': 'δΔ', 'epsilon': 'εΕϵ', 'zeta': 'ζΖ', 'eta': 'ηΗ',
    'theta': 'θΘϑ', 'iota': 'ιΙ', 'kappa': 'κΚϰ', 'lambda': 'λΛ', 'mu': 'μΜµ', 'nu': 'νΝ', 'xi': 'ξΞ',
    'omicron': 'οΟ', 'pi': 'πΠϖ', 'rho': 'ρΡϱ', 'sigma': 'σςΣ', 'tau': 'τΤ', 'upsilon': 'υΥ', 'phi': 'φΦϕ',
    'chi': 'χΧ', 'psi': 'ψΨ', 'omega': 'ωΩ',
}
GREEK = {ch: name for name, chars in NAMES.items() for ch in chars}


def greek_to_names(text):
    """Replaces every Greek letter by its name, as a separate word (see the module docstring)."""
    out = []
    for i, ch in enumerate(text):
        name = GREEK.get(ch)
        if name is None:
            out.append(ch)
            continue
        before = text[i - 1] if i > 0 else ''
        after = text[i + 1] if i + 1 < len(text) else ''
        # A Greek letter before this one already added its trailing space.
        if before.isalnum() and before not in GREEK:
            name = ' ' + name
        if after.isalnum():
            name += ' '
        out.append(name)
    return ''.join(out)


if __name__ == '__main__':
    examples = {'TNF-α': 'TNF-alpha', 'α-synuclein': 'alpha-synuclein', 'IFNγ': 'IFN gamma',
                'TGFβ1': 'TGF beta 1', '10 μM': '10 mu M', '5 µg': '5 mu g', 'αβ T cells': 'alpha beta T cells',
                'ΔNp63': 'delta Np63', 'NF-κB': 'NF-kappa B', 'no Greek here': 'no Greek here'}
    wrong = {k: greek_to_names(k) for k, v in examples.items() if greek_to_names(k) != v}
    print('all examples correct' if not wrong else f'WRONG: {wrong}')
    raise SystemExit(1 if wrong else 0)
