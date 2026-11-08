#!/usr/bin/env python3
"""Google Play release for Orch Neural Network (com.johnseong.orchneuralnetwork).

The app must first exist in Play Console (created by hand) and the service account must have access.

    python3 tools/play_release.py check        # can the service account see the app?
    python3 tools/play_release.py listing      # en-GB texts, icon, feature graphic, screenshots
    python3 tools/play_release.py internal     # upload the AAB to the internal track
    python3 tools/play_release.py production   # promote that versionCode to production
    python3 tools/play_release.py production draft   # while the app is still a draft in Play Console
    python3 tools/play_release.py products     # tip jar in-app products (needs an uploaded AAB with billing first)

Key: PLAY_SA_JSON (default ~/Documents/keys/havit-play.json). Listing assets come from PLAY_DIR
(default ~/Documents/sr-video/screens/orchnn/play). Never prints keys.
"""
import base64, glob, json, os, subprocess, sys, tempfile, time, urllib.error, urllib.parse, urllib.request

PKG = "com.johnseong.orchneuralnetwork"
LANG = "en-GB"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAY = os.path.expanduser(os.environ.get("PLAY_DIR", "~/Documents/sr-video/screens/orchnn/play"))
# The signed AAB is kept beside the listing (build outputs get cleaned); fall back to the Gradle output.
AAB = next((p for p in (os.environ.get("AAB"), os.path.join(PLAY, "orchnn-1.0-release.aab"),
                        os.path.join(ROOT, "app", "build", "outputs", "bundle", "release", "app-release.aab"))
            if p and os.path.exists(p)), os.path.join(PLAY, "orchnn-1.0-release.aab"))
API = "https://androidpublisher.googleapis.com/androidpublisher/v3/applications/" + PKG
UPLOAD = "https://androidpublisher.googleapis.com/upload/androidpublisher/v3/applications/" + PKG
NOTES = ("First Android release. Type a sentence and watch a real 82-million-parameter GPT-2 predict your next word on your "
         "device: attention heads, neurons, the logit lens and every weight. Optional tip jar in About.")
VERSION = "1.0"


def token():
    k = json.load(open(os.path.expanduser(os.environ.get("PLAY_SA_JSON", "~/Documents/keys/havit-play.json"))))
    b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=")
    now = int(time.time())
    h = b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    c = b64(json.dumps({"iss": k["client_email"], "scope": "https://www.googleapis.com/auth/androidpublisher",
                        "aud": k["token_uri"], "iat": now, "exp": now + 3600}).encode())
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(k["private_key"])
        pk = f.name
    try:
        sig = subprocess.run(["openssl", "dgst", "-sha256", "-sign", pk], input=h + b"." + c, capture_output=True).stdout
    finally:
        os.remove(pk)
    body = urllib.parse.urlencode({"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                                   "assertion": (h + b"." + c + b"." + b64(sig)).decode()}).encode()
    return json.load(urllib.request.urlopen(k["token_uri"], body))["access_token"]


TOK = None


def call(url, method="GET", data=None, raw=None, ctype="application/json"):
    global TOK
    TOK = TOK or token()
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Authorization": "Bearer " + TOK, "Content-Type": ctype})
    try:
        r = urllib.request.urlopen(req, timeout=900).read()
        return json.loads(r) if r else {}
    except urllib.error.HTTPError as e:
        return {"err": e.code, "body": e.read().decode()[:800]}


def edit():
    e = call(API + "/edits", "POST", {})
    if "id" not in e:
        sys.exit(f"cannot open an edit for {PKG}: {e}")
    return e["id"]


def commit(eid):
    print("commit", call(API + f"/edits/{eid}:commit", "POST", {}))


def text(name):
    return open(os.path.join(PLAY, name)).read().strip()


def listing():
    eid = edit()
    for t in (NOTES, text("short-description.txt"), text("full-description.txt")):
        assert "·" not in t and "•" not in t
    print("details", call(API + f"/edits/{eid}/details", "PATCH", {"defaultLanguage": LANG, "contactEmail": "john@orchestrsim.com"}))
    print("listing", call(API + f"/edits/{eid}/listings/{LANG}", "PUT", {
        "language": LANG, "title": text("title.txt"),
        "shortDescription": text("short-description.txt"), "fullDescription": text("full-description.txt")}).get("title"))
    imgs = {"icon": ["play-icon-512.png"], "featureGraphic": ["feature-graphic-1024x500.png"],
            "phoneScreenshots": sorted(glob.glob(os.path.join(PLAY, "screenshots", "phone", "*.png"))),
            "sevenInchScreenshots": sorted(glob.glob(os.path.join(PLAY, "screenshots", "tablet7", "*.png"))),
            "tenInchScreenshots": sorted(glob.glob(os.path.join(PLAY, "screenshots", "tablet10", "*.png")))}
    for kind, files in imgs.items():
        call(API + f"/edits/{eid}/listings/{LANG}/{kind}", "DELETE")
        for f in files:
            r = call(UPLOAD + f"/edits/{eid}/listings/{LANG}/{kind}?uploadType=media", "POST",
                     raw=open(os.path.join(PLAY, f), "rb").read(), ctype="image/png")
            print(kind, os.path.basename(f), "ok" if "image" in r else r)
    commit(eid)


def internal():
    if not os.path.exists(AAB):
        sys.exit("Build the signed AAB first: ./gradlew :app:bundleRelease")
    eid = edit()
    up = call(UPLOAD + f"/edits/{eid}/bundles?uploadType=media", "POST", raw=open(AAB, "rb").read(),
              ctype="application/octet-stream")
    if "versionCode" not in up:
        sys.exit(f"upload failed: {up}")
    vc = up["versionCode"]
    print("uploaded versionCode", vc)
    print("track", call(API + f"/edits/{eid}/tracks/internal", "PUT", {"track": "internal", "releases": [
        {"name": f"{VERSION} ({vc})", "versionCodes": [str(vc)], "status": "completed",
         "releaseNotes": [{"language": LANG, "text": NOTES}]}]}).get("track"))
    commit(eid)


def production():
    eid = edit()
    rel = call(API + f"/edits/{eid}/tracks/internal")
    vcs = rel["releases"][0]["versionCodes"]
    print("track", call(API + f"/edits/{eid}/tracks/production", "PUT", {"track": "production", "releases": [
        {"name": f"{VERSION} ({vcs[0]})", "versionCodes": vcs, "status": sys.argv[2] if len(sys.argv) > 2 else "completed",
         "releaseNotes": [{"language": LANG, "text": NOTES}]}]}))
    commit(eid)


# Tip jar: managed products the app consumes after purchase, so they unlock nothing and can be bought again.
# Prices are USD (units, nanos); Play converts the other countries. Ids/names/descriptions match the App Store.
TIPS = [("orchnn.tip.small", "Small tip", "A small thank-you. Unlocks nothing.", (1, 990000000)),
        ("orchnn.tip.medium", "Medium tip", "A medium thank-you. Unlocks nothing.", (4, 990000000)),
        ("orchnn.tip.large", "Large tip", "A large thank-you. Unlocks nothing.", (9, 990000000))]


def money(m):
    return {k: m[k] for k in ("currencyCode", "units", "nanos") if k in m}


def products():
    """One-time products via monetization.onetimeproducts (the legacy inappproducts API is refused).
    US price is set exactly; other regions use Play's own conversion of the USD price."""
    for sku, title, desc, usd in TIPS:
        usdm = {"currencyCode": "USD", "units": str(usd[0]), "nanos": usd[1]}
        conv = call(API + "/pricing:convertRegionPrices", "POST", {"price": usdm})
        if "convertedRegionPrices" not in conv:
            sys.exit(f"price conversion failed: {conv}")
        regions = [{"regionCode": rc, "price": money(v["price"]), "availability": "AVAILABLE"}
                   for rc, v in conv["convertedRegionPrices"].items() if rc != "US"]
        regions.append({"regionCode": "US", "price": usdm, "availability": "AVAILABLE"})
        other = conv.get("convertedOtherRegionsPrice", {})
        opt = {"purchaseOptionId": "tip", "buyOption": {"legacyCompatible": True, "multiQuantityEnabled": False},
               "regionalPricingAndAvailabilityConfigs": regions}
        if other.get("usdPrice") and other.get("eurPrice"):
            opt["newRegionsConfig"] = {"usdPrice": money(other["usdPrice"]), "eurPrice": money(other["eurPrice"]),
                                       "availability": "AVAILABLE"}
        body = {"packageName": PKG, "productId": sku,
                "listings": [{"languageCode": loc, "title": title, "description": desc} for loc in (LANG, "en-US")],
                "purchaseOptions": [opt]}
        q = "?allowMissing=true&updateMask=listings,purchaseOptions&regionsVersion.version=2026/01"
        r = call(API + f"/onetimeproducts/{sku}{q}", "PATCH", body)
        if r.get("productId") != sku:
            print(sku, "FAILED", r); continue
        a = call(API + f"/oneTimeProducts/{sku}/purchaseOptions:batchUpdateStates", "POST", {"requests": [
            {"activatePurchaseOptionRequest": {"packageName": PKG, "productId": sku, "purchaseOptionId": "tip"}}]})
        state = [o.get("state") for p in a.get("oneTimeProducts", []) for o in p.get("purchaseOptions", [])]
        print(sku, f"{len(regions)} regions", "state", state or a)


def check():
    e = call(API + "/edits", "POST", {})
    if "id" in e:
        print("OK: service account can edit", PKG)
        call(API + f"/edits/{e['id']}", "DELETE")
    else:
        print("NO ACCESS:", e)


if __name__ == "__main__":
    {"check": check, "listing": listing, "internal": internal, "production": production, "products": products}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
