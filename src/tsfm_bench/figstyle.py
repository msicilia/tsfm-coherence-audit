"""Shared matplotlib style for the manuscript figures.

Figures are drawn at the width of the elsarticle text block (about 4.7 in in
review mode) and included at \\linewidth, so the type prints at its nominal size.
Text is set in Latin Modern Roman, the face of the manuscript body, and embedded
as TrueType (fonttype 42) rather than Type 3, which Elsevier rejects. TeX Live
ships Latin Modern as CFF OpenType, which matplotlib cannot embed cleanly as
TrueType, so the four 10 pt faces are converted to TrueType once (cubic to
quadratic outlines, fontTools) and cached under .cache/fonts. Without TeX Live
or fontTools the figures fall back to the DejaVu Serif face that matplotlib
bundles. Titles are left to the captions.
"""
from __future__ import annotations

import glob
import os
import subprocess

from .paths import ROOT

TEXT_WIDTH_IN = 4.7
FONT_CACHE = ROOT / ".cache" / "fonts"

# Categorical slots in fixed order (never cycled) and a single-hue sequential
# ramp, light -> dark, from a colour-blind-validated reference palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_MUTED = "#0b0b0b", "#52514e"


def _otf_to_ttf(src: str, dst: str, max_err: float = 1.0) -> None:
    """Convert a CFF OpenType font to TrueType outlines (after fontmake's otf2ttf)."""
    from fontTools.ttLib import TTFont, newTable
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    font = TTFont(src)
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, max_err, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    hmtx = font["hmtx"]
    for name, glyph in glyf.glyphs.items():
        if hasattr(glyph, "xMin"):
            hmtx[name] = (hmtx[name][0], glyph.xMin)
    maxp = font["maxp"] = newTable("maxp")
    for key, value in dict(tableVersion=0x00010000, maxZones=1, maxTwilightPoints=0,
                           maxStorage=0, maxFunctionDefs=0, maxInstructionDefs=0,
                           maxStackElements=0, maxSizeOfInstructions=0,
                           maxComponentElements=0).items():
        setattr(maxp, key, value)
    maxp.numGlyphs = len(order)
    maxp.recalc(font)
    maxp.compile(font)
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    font.sfntVersion = "\000\001\000\000"
    font.recalcBBoxes = True
    font.save(dst)


def _latin_modern() -> str | None:
    """Register TrueType Latin Modern Roman, converting from TeX Live's OTF once."""
    try:
        regular = subprocess.run(["kpsewhich", "lmroman10-regular.otf"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        regular = ""
    if not regular:
        return None
    try:
        FONT_CACHE.mkdir(parents=True, exist_ok=True)
        ttfs = []
        for src in glob.glob(os.path.join(os.path.dirname(regular), "lmroman10-*.otf")):
            dst = FONT_CACHE / (os.path.basename(src)[:-4] + ".ttf")
            if not dst.exists():
                _otf_to_ttf(src, str(dst))
            ttfs.append(str(dst))
        from matplotlib import font_manager
        for f in ttfs:
            font_manager.fontManager.addfont(f)
    except Exception as e:  # fontTools missing or conversion failed
        print(f"  [figstyle] Latin Modern unavailable ({type(e).__name__}); using DejaVu Serif")
        return None
    return "Latin Modern Roman"


def apply() -> None:
    import matplotlib.pyplot as plt
    family = _latin_modern() or "DejaVu Serif"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [family],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.grid": True, "grid.alpha": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "cm",
    })
