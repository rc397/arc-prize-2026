"""Render paper/card.svg to a 560x280 PNG for the Kaggle thumbnail upload. Tries cairosvg, then
svglib+reportlab."""
import os
SVG = r"D:\arc\paper\card.svg"
PNG = r"D:\arc\paper\card.png"
ok = False
try:
    import cairosvg
    cairosvg.svg2png(url=SVG, write_to=PNG, output_width=560, output_height=280)
    print("OK cairosvg", flush=True)
    ok = True
except Exception as e1:
    print("cairosvg unavailable:", repr(e1)[:90], flush=True)
if not ok:
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        renderPM.drawToFile(svg2rlg(SVG), PNG, fmt="PNG")
        print("OK svglib", flush=True)
        ok = True
    except Exception as e2:
        print("svglib unavailable:", repr(e2)[:90], flush=True)
print("PNG exists:", os.path.exists(PNG), "bytes:", os.path.getsize(PNG) if os.path.exists(PNG) else 0,
      flush=True)
