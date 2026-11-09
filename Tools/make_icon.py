"""Inside LLM app icon (2026-11): attention arcs drawn as glowing points, in the Point Fluid style.
Concepts and the contact sheet live in Tools/icon_concepts.py (concept_a2 is the shipped one).

    python3 Tools/make_icon.py [PLAY_DIR]
writes the iOS (light/dark/tinted), macOS, Android adaptive/monochrome/legacy icons and,
when PLAY_DIR is given, play-icon-512.png there.
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icon_concepts as ic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    c_holder = {}
    orig_render = ic.Canvas.render
    def render(self):
        c_holder["dots"] = self.dots.resize((1024, 1024), Image.LANCZOS)
        return orig_render(self)
    ic.Canvas.render = render
    full, layer = ic.concept_a2()
    dots = c_holder["dots"]

    # iOS: light = full tile, dark = same (already dark), tinted = greyscale points on black
    ios = os.path.join(ROOT, "OrchNeuralNetwork/Resources/Assets.xcassets/AppIcon.appiconset")
    for f in os.listdir(ios):
        if f.endswith(".png"): os.remove(os.path.join(ios, f))
    full.save(os.path.join(ios, "AppIcon.png"))
    full.save(os.path.join(ios, "AppIcon-dark.png"))
    grey = ImageOps.grayscale(layer.convert("RGB")).point(lambda v: min(255, int(v * 1.25)))
    black = Image.new("L", (1024, 1024), 0); black.paste(grey, (0, 0), layer.split()[3])
    black.convert("RGB").save(os.path.join(ios, "AppIcon-tinted.png"))
    base = {"idiom": "universal", "platform": "ios", "size": "1024x1024"}
    json.dump({"images": [dict(base, filename="AppIcon.png"),
                          dict(base, filename="AppIcon-dark.png", appearances=[{"appearance": "luminosity", "value": "dark"}]),
                          dict(base, filename="AppIcon-tinted.png", appearances=[{"appearance": "luminosity", "value": "tinted"}])],
               "info": {"author": "xcode", "version": 1}}, open(os.path.join(ios, "Contents.json"), "w"), indent=2)

    # macOS: rounded tile with margin and soft shadow (same recipe as Point Fluid)
    tile = 824
    mask = Image.new("L", (tile * 2, tile * 2), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, tile * 2 - 1, tile * 2 - 1], radius=370, fill=255)
    mask = mask.resize((tile, tile), Image.LANCZOS)
    mac = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    sh = Image.new("L", (1024, 1024), 0); sh.paste(mask, (100, 112))
    shadow = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    shadow.putalpha(sh.filter(ImageFilter.GaussianBlur(14)).point(lambda a: int(a * 0.45)))
    mac = Image.alpha_composite(mac, shadow)
    face = full.resize((tile, tile), Image.LANCZOS).convert("RGBA"); face.putalpha(mask)
    mac.alpha_composite(face, (100, 100))
    macdir = os.path.join(ROOT, "Mac/MacAssets.xcassets/MacAppIcon.appiconset")
    for size in (16, 32, 128, 256, 512):
        for s in (1, 2):
            mac.resize((size * s, size * s), Image.LANCZOS).save(os.path.join(macdir, f"icon_{size}x{size}@{s}x.png"))

    # Android adaptive icon: 108dp layers, art (1024 tile) mapped to the central 72dp
    res = os.path.join(ROOT, "android/app/src/main/res")
    dens = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}
    for name, k in dens.items():
        L = int(108 * k); A = int(72 * k); off = (L - A) // 2
        fg = Image.new("RGBA", (L, L), (0, 0, 0, 0)); fg.alpha_composite(layer.resize((A, A), Image.LANCZOS), (off, off))
        fg.save(os.path.join(res, f"drawable-{name}/ic_launcher_foreground.png"))
        mono = Image.new("RGBA", (L, L), (0, 0, 0, 0))
        a = dots.split()[3].resize((A, A), Image.LANCZOS)
        m = Image.new("RGBA", (A, A), (255, 255, 255, 0)); m.putalpha(a); mono.alpha_composite(m, (off, off))
        mono.save(os.path.join(res, f"drawable-{name}/ic_launcher_monochrome.png"))
        ic.background(L).save(os.path.join(res, f"drawable-{name}/ic_launcher_background.png"))
        full.resize((int(48 * k), int(48 * k)), Image.LANCZOS).save(os.path.join(res, f"mipmap-{name}/ic_launcher.png"))
    open(os.path.join(res, "mipmap-anydpi-v26/ic_launcher.xml"), "w").write(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n'
        '    <background android:drawable="@drawable/ic_launcher_background" />\n'
        '    <foreground android:drawable="@drawable/ic_launcher_foreground" />\n'
        '    <monochrome android:drawable="@drawable/ic_launcher_monochrome" />\n'
        '</adaptive-icon>\n')

    if len(sys.argv) > 1:
        full.resize((512, 512), Image.LANCZOS).save(os.path.join(os.path.expanduser(sys.argv[1]), "play-icon-512.png"))
    print("ok")

if __name__ == "__main__":
    main()
