# App Store submission kit — Orch Neural Network

## Identity
- Bundle ID: `com.johnseong.OrchNeuralNetwork`
- Team: Z64KRUX3W3 (automatic signing)
- Version 1.0 (build 1), iOS 17.0+, iPhone and iPad
- Primary category: Education · Secondary: Developer Tools
- Age rating: 4+ (no objectionable content; user-typed text is processed on device only)
- Price: Free

## Name / subtitle (30 chars each)
- Name: `Orch Neural Network`
- Subtitle: `See inside a language model`

## Promotional text (170 chars)
Type a sentence and watch a real 82-million-parameter GPT-2 predict your next word, layer by layer, entirely on your device.

## Description
Ever wondered what your keyboard is doing when it suggests the next word? Orch Neural Network puts a real language model on your phone and lets you watch it think.

Type anything. A six-layer GPT-2 runs on your device after every keystroke and shows:

• Suggestions — the three most probable next words, exactly as a predictive keyboard would show them, with their real probabilities.
• Signal through the network — six layers lighting up as the forward pass runs.
• Attention — a heatmap of which earlier words each token is reading from, for any of the 72 attention heads, or averaged.
• Neurons — all 3,072 MLP neurons in each layer, firing for your last word.
• Logit lens — what the model would guess if it stopped after layer 0, 1, 2… and how that guess sharpens or changes as it goes deeper.
• Weights — every one of the model's 76 weight matrices, rendered as a zoomable heatmap of the actual numbers learned in training.
• Learn — a six-step walkthrough of one forward pass (tokenize, embed, attend, MLP, stack, softmax) using your own words as the example.

Everything runs locally. No account, no network, no analytics. Nothing you type leaves the device.

The model is DistilGPT2 (Hugging Face, Apache 2.0), a distilled version of OpenAI's GPT-2 trained on 2019 web text. Its suggestions are shown for learning purposes and can be wrong or dated.

## Keywords (100 chars)
neural network,LLM,GPT,transformer,attention,machine learning,AI,visualizer,keyboard,education

## What's new (1.0)
First release.

## URLs
- Support URL: TODO (e.g. https://orchaerospace.com/support)
- Privacy policy URL: TODO — required by App Store Connect even with no data collection. One paragraph stating no data is collected is sufficient.
- Marketing URL: optional

## App Privacy (App Store Connect questionnaire)
- Data collection: **No, we do not collect data from this app.**
- PrivacyInfo.xcprivacy is bundled: no tracking, no collected data, UserDefaults reason CA92.1.

## Export compliance
`ITSAppUsesNonExemptEncryption = false` is set in Info.plist, so no encryption documentation is needed.

## Review notes (paste into "Notes" for App Review)
The app bundles a small open-source language model (DistilGPT2, Apache 2.0) and runs it fully offline. Text typed by the user is used only to compute next-word predictions on device. No sign-in is needed. Suggested test: tap the "I am going…" sample button on the Predict tab, then open the Inside tab and tap the play button.

## Screenshots to capture (6.9" and 6.5" iPhone, 13" iPad)
1. Predict tab with "I am going to the " entered — suggestion bar and probability bars visible.
2. Inside tab — layer stack with logit-lens arrows.
3. Inside tab scrolled to the attention heatmap, a specific head selected.
4. Weights tab — a layer's Q·K·V projection matrix zoomed in.
5. Learn tab — step 5 (logit lens table).
Capture from the simulator: `xcrun simctl io booted screenshot shot.png` after launching with `-prefill "I am going to the "`.

## Build and upload
```bash
xcodegen generate                      # regenerate the .xcodeproj if project.yml changed
open OrchNeuralNetwork.xcodeproj       # Product ▸ Archive ▸ Distribute App ▸ App Store Connect
```
or from the CLI:
```bash
xcodebuild -project OrchNeuralNetwork.xcodeproj -scheme OrchNeuralNetwork -configuration Release \
  -destination 'generic/platform=iOS' -archivePath build/OrchNeuralNetwork.xcarchive archive
xcodebuild -exportArchive -archivePath build/OrchNeuralNetwork.xcarchive \
  -exportOptionsPlist ExportOptions.plist -exportPath build/export
```

## Pre-flight checklist
- [ ] Create the app record in App Store Connect with the bundle ID above.
- [ ] Add privacy policy URL and support URL.
- [ ] Upload screenshots for 6.9", 6.5" iPhone and 13" iPad.
- [ ] Archive with a Release build, upload, select the build, submit.
