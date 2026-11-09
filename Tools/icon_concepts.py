"""Inside LLM icon concepts in the Point Fluid (Eddyfield) visual language:
navy gradient tile, glowing turbo-coloured points, lots of negative space.
    python3 make_concepts.py  ->  concept-{a,b,c,d}.png + concepts.png
"""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.dirname(os.path.abspath(__file__))
S = 3
N = 1024 * S

def turbo(x):
    x = min(max(x, 0.0), 1.0)
    r = 0.13572138 + x*(4.61539260 + x*(-42.66032258 + x*(132.13108234 + x*(-152.94239396 + x*59.28637943))))
    g = 0.09140261 + x*(2.19418839 + x*(4.84296658 + x*(-14.18503333 + x*(4.27729857 + x*2.82956604))))
    b = 0.10667330 + x*(12.64194608 + x*(-60.58204836 + x*(110.36276771 + x*(-89.90310912 + x*27.34824973))))
    return tuple(int(255*min(max(c, 0), 1)) for c in (r, g, b))

def background(n=N):
    y = np.linspace(0, 1, n)[:, None]; x = np.linspace(0, 1, n)[None, :]
    top = np.array([0x07, 0x14, 0x2B]); bot = np.array([0x0B, 0x3A, 0x63])
    t = np.clip(0.75*y + 0.25*x, 0, 1)[..., None]
    return Image.fromarray((top*(1-t) + bot*t).astype(np.uint8), "RGB")

class Canvas:
    def __init__(self):
        self.dots = Image.new("RGBA", (N, N), (0, 0, 0, 0)); self.glow = Image.new("RGBA", (N, N), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.dots); self.g = ImageDraw.Draw(self.glow)
    def dot(self, x, y, r, col, glow=2.2, ga=120):
        x, y, r = x*N, y*N, r*N
        self.d.ellipse([x-r, y-r, x+r, y+r], fill=col)
        R = r*glow; self.g.ellipse([x-R, y-R, x+R, y+R], fill=col + (ga,))
    def chain(self, pts, spacing, rad, colf, skip_ends=0.0):
        """Dots at equal arc spacing along a dense polyline; rad/colf take t in [0,1]."""
        pts = np.asarray(pts); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = seg.sum(); cum = np.concatenate([[0], np.cumsum(seg)])
        n = max(int(L / spacing), 1)
        for k in range(n + 1):
            s = L * k / n; t = s / L
            if t < skip_ends or t > 1 - skip_ends: continue
            i = min(np.searchsorted(cum, s) - 1, len(seg) - 1); i = max(i, 0)
            f = (s - cum[i]) / seg[i] if seg[i] else 0
            p = pts[i] + f * (pts[i+1] - pts[i])
            self.dot(p[0], p[1], rad(t) if callable(rad) else rad, colf(t))
    def render(self):
        glow = self.glow.filter(ImageFilter.GaussianBlur(10 * S))
        layer = Image.alpha_composite(glow, self.dots)
        img = Image.alpha_composite(background().convert("RGBA"), layer).convert("RGB")
        return img.resize((1024, 1024), Image.LANCZOS), layer.resize((1024, 1024), Image.LANCZOS)

def arc(a, b, h, n=400):
    """Semicircle-ish arc from a to b bulging by h (negative = up)."""
    t = np.linspace(0, 1, n)
    x = a[0] + (b[0]-a[0])*t
    y = a[1] + (b[1]-a[1])*t + h*np.sin(np.pi*t)
    return np.stack([x, y], 1)

# A: attention arcs. A row of tokens; every earlier token sends a dotted arc to the last one.
def concept_a():
    c = Canvas(); y0 = 0.70; xs = np.linspace(0.14, 0.86, 6); last = xs[-1]
    w = [0.08, 0.22, 0.12, 0.45, 0.30]          # attention weights to the last token
    for x, a in zip(xs[:-1], w):
        h = -(last - x) * 0.78
        col = turbo(0.18 + 0.72*a/max(w))
        c.chain(arc((x, y0), (last, y0), h), 0.030, lambda t, a=a: 0.0085 + 0.006*a/max(w), lambda t, col=col: col, skip_ends=0.04)
    for i, x in enumerate(xs):
        c.dot(x, y0, 0.026 if i == len(xs)-1 else 0.019, turbo(0.86) if i == len(xs)-1 else turbo(0.25), ga=150)
    return c.render()

# B: causal attention matrix as a sparse dot grid (lower triangle), hot diagonal + one strong head pattern.
def concept_b():
    c = Canvas(); n = 6; m = 0.19; step = (1 - 2*m) / (n - 1)
    rng = np.random.default_rng(3)
    for i in range(n):
        for j in range(i + 1):
            a = 1.0 if j == i else (0.75 if j == 0 else (0.55 if j == i-1 else 0.15 + 0.1*rng.random()))
            c.dot(m + j*step, m + i*step, 0.010 + 0.024*a, turbo(0.15 + 0.75*a), ga=110)
    return c.render()

# C: residual stream. Token streamlines enter at the left, converge through layers into one hot point,
#    and a single stream leaves to the right: many tokens in, one next token out.
def concept_c():
    c = Canvas(); focus = np.array([0.66, 0.52]); k = 6
    for i in range(k):
        y = 0.16 + 0.68*i/(k-1); t = np.linspace(0, 1, 500)
        x = 0.0 + (focus[0]-0.0)*t
        yy = y + (focus[1]-y)*(3*t**2 - 2*t**3)
        c.chain(np.stack([x, yy], 1), 0.032, lambda t: 0.0145 - 0.007*t, lambda t: turbo(0.15 + 0.55*t), skip_ends=0.0)
    t = np.linspace(0, 1, 100)
    c.chain(np.stack([focus[0] + 0.40*t, np.full_like(t, focus[1])], 1), 0.036, lambda t: 0.012 + 0.006*t, lambda t: turbo(0.80 + 0.08*t))
    c.dot(*focus, 0.030, turbo(0.86), ga=170)
    return c.render()

# D: eye into the model. Attention arcs above and below a token row form an almond; the predicted token is the iris.
def concept_d():
    c = Canvas(); y0 = 0.5; L, R = 0.10, 0.90
    for h, v in [(-0.30, 0.30), (-0.20, 0.45), (0.30, 0.30), (0.20, 0.45)]:
        c.chain(arc((L, y0), (R, y0), h), 0.030, 0.0095, lambda t, v=v: turbo(v + 0.15*math.sin(math.pi*t)), skip_ends=0.02)
    xs = np.linspace(L, R, 9)
    for x in xs:
        if abs(x-0.5) < 1e-6: continue
        c.dot(x, y0, 0.011, turbo(0.22), ga=90)
    c.dot(0.5, y0, 0.075, turbo(0.86), glow=1.6, ga=170)
    return c.render()

def preview_sheet(images, labels):
    from PIL import ImageFont
    W = 60+120+180 + 4*30 + 340; rowH = 2*(180+40) + 20
    sheet = Image.new("RGB", (W, rowH*len(images) + 20), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception: font = None
    for r, (im, lab) in enumerate(zip(images, labels)):
        y = 20 + r*rowH
        # dark band behind the second preview row
        d.rectangle([0, y + 220, W, y + 440], fill=(24, 24, 28))
        sheet.paste(im.resize((300, 300), Image.LANCZOS), (20, y + 60))
        d.text((20, y + 20), lab, fill=(20, 20, 20), font=font)
        for band, by in [(0, y + 20), (1, y + 240)]:
            x = 360
            for px in (60, 120, 180):
                m = Image.new("L", (px*4, px*4), 0)
                ImageDraw.Draw(m).rounded_rectangle([0, 0, px*4-1, px*4-1], radius=int(px*4*0.2237), fill=255)
                ic = im.resize((px, px), Image.LANCZOS); ic.putalpha(m.resize((px, px), Image.LANCZOS))
                sheet.paste(ic, (x, by + (180-px)//2), ic); x += px + 30
    return sheet

if __name__ == "__main__" and "a2" not in sys.argv:
    outs = {}
    for key, fn, lab in [("a", concept_a, "A  Attention arcs"), ("b", concept_b, "B  Causal attention matrix"),
                         ("c", concept_c, "C  Residual stream to next token"), ("d", concept_d, "D  Eye of attention")]:
        img, layer = fn(); img.save(os.path.join(OUT, f"concept-{key}.png")); layer.save(os.path.join(OUT, f"concept-{key}-points.png")); outs[key] = (img, lab)
    preview_sheet([v[0] for v in outs.values()], [v[1] for v in outs.values()]).save(os.path.join(OUT, "concepts.png"))
    print("ok")

# A2 (chosen): attention arcs, rebalanced. Token row low-centre, nested arcs fill the tile,
# heavier points so the arcs survive at 60 px.
def concept_a2():
    c = Canvas(); y0 = 0.73; xs = np.linspace(0.13, 0.87, 6); last = xs[-1]
    w = [0.35, 0.10, 0.20, 0.55, 0.85]
    for x, a in zip(xs[:-1], w):
        h = -(last - x) * 0.74
        col = turbo(0.16 + 0.74*a)
        c.chain(arc((x, y0), (last, y0), h), 0.033, 0.0105 + 0.006*a, lambda t, col=col: col, skip_ends=0.045)
    for i, x in enumerate(xs[:-1]):
        c.dot(x, y0, 0.021, turbo(0.22), ga=150)
    c.dot(last, y0, 0.032, turbo(0.90), ga=180)
    return c.render()

if __name__ == "__main__" and "a2" in sys.argv:
    img, layer = concept_a2(); img.save(os.path.join(OUT, "concept-a2.png")); layer.save(os.path.join(OUT, "concept-a2-points.png"))
    ims = [Image.open(os.path.join(OUT, f"concept-{k}.png")) for k in "abcd"] + [img]
    preview_sheet(ims, ["A  Attention arcs", "B  Causal attention matrix", "C  Residual stream to next token", "D  Eye of attention", "A2  Attention arcs, refined (chosen)"]).save(os.path.join(OUT, "concepts.png"))
