#if os(macOS)
import SwiftUI

/// Window state shared by the sidebar and the menu bar commands.
@Observable
final class MacNavigation {
    enum Section: Int, CaseIterable, Identifiable {
        case predict, inside, weights, steps
        var id: Int { rawValue }
        var title: String { ["Predict", "Inside", "Weights", "Steps"][rawValue] }
        var glyph: String { ["keyboard", "brain.head.profile", "square.grid.3x3.fill", "list.number"][rawValue] }
        var key: KeyEquivalent { KeyEquivalent(Character("\(rawValue + 1)")) }
    }
    var section: Section? = Section(rawValue: UserDefaults.standard.integer(forKey: "tab")) ?? .predict
    var showAbout = UserDefaults.standard.bool(forKey: "showAbout")
    var showTipJar = false
}

/// Mac layout: a resizable window with a sidebar of the four sections and the model's vital signs.
struct MacContentView: View {
    @Environment(InferenceEngine.self) private var engine
    @Environment(MacNavigation.self) private var nav
    /// Screenshot mode (`-shotWindow 1`, debug) collapses the sidebar, which offscreen rendering cannot draw.
    @State private var columns: NavigationSplitViewVisibility = UserDefaults.standard.bool(forKey: "shotWindow") ? .detailOnly : .automatic

    var body: some View {
        @Bindable var nav = nav
        NavigationSplitView(columnVisibility: $columns) {
            List(selection: $nav.section) {
                Section("Orch Neural Network") {
                    ForEach(MacNavigation.Section.allCases) { s in
                        Label(s.title, systemImage: s.glyph).tag(s)
                    }
                }
                if let c = engine.config {
                    Section("Model") {
                        LabeledContent("Layers", value: "\(c.n_layer)")
                        LabeledContent("Heads", value: "\(c.n_head)")
                        LabeledContent("Width", value: "\(c.n_embd)")
                        LabeledContent("Parameters", value: engine.parameterCount.grouped)
                        if let t = engine.trace {
                            LabeledContent("Last pass", value: "\(Int(t.elapsedMilliseconds)) ms")
                        }
                    }
                    .font(.caption)
                }
                Section {
                    Button { nav.showAbout = true } label: { Label("About and tip jar", systemImage: "info.circle") }
                        .buttonStyle(.plain)
                }
            }
            .scrollContentBackground(.hidden)
            .background(Theme.sheet)
            .navigationSplitViewColumnWidth(min: 200, ideal: 230, max: 300)
        } detail: {
            Group {
                switch nav.section ?? .predict {
                case .predict: PredictView()
                case .inside: InsideView()
                case .weights: WeightsView()
                case .steps: LearnView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .tint(Theme.red)
        .overlay { MacLoadingOverlay() }
        .onAppear {
            engine.load()
            #if DEBUG
            // `-shotWindow 1`: fixed 1280-point-wide window for App Store screenshots.
            if UserDefaults.standard.bool(forKey: "shotWindow") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    NSApp.windows.first?.setContentSize(NSSize(width: 1280, height: 852))
                    NSApp.windows.first?.center()
                }
            }
            // `-shotOut /path.png`: renders the whole window (frame view) to a PNG after the model has run, then quits.
            if let out = UserDefaults.standard.string(forKey: "shotOut") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 12) {
                    guard let frame = NSApp.windows.first?.contentView?.superview,
                          let rep = frame.bitmapImageRepForCachingDisplay(in: frame.bounds) else { NSApp.terminate(nil); return }
                    frame.cacheDisplay(in: frame.bounds, to: rep)
                    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: out))
                    NSApp.terminate(nil)
                }
            }
            #endif
        }
        .sheet(isPresented: $nav.showAbout) { AboutView().macSheetFrame() }
    }
}

private struct MacLoadingOverlay: View {
    @Environment(InferenceEngine.self) private var engine
    var body: some View {
        switch engine.state {
        case .ready: EmptyView()
        case .loading:
            ZStack {
                Theme.paper
                VStack(spacing: 14) {
                    ProgressView().controlSize(.large)
                    Text("Loading 82 million weights…").font(Theme.display(18, relativeTo: .headline))
                    Text("Everything runs on this Mac. Nothing leaves it.").font(.footnote).foregroundStyle(Theme.muted)
                }
            }
        case .failed(let message):
            ZStack {
                Theme.paper
                VStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle").font(.largeTitle)
                    Text("Couldn't load the model").font(.headline)
                    Text(message).font(.footnote).foregroundStyle(Theme.muted)
                }
            }
        }
    }
}

/// Menu bar: section shortcuts (Command-1 to 4), Run Now (Command-R), sample sentences, About.
struct MacCommands: Commands {
    let nav: MacNavigation
    let engine: InferenceEngine

    var body: some Commands {
        CommandGroup(replacing: .appInfo) {
            Button("About Orch Neural Network") { nav.showAbout = true }
        }
        CommandGroup(after: .sidebar) {
            ForEach(MacNavigation.Section.allCases) { s in
                Button(s.title) { nav.section = s }.keyboardShortcut(s.key, modifiers: .command)
            }
            Divider()
        }
        CommandMenu("Model") {
            Button("Run Now") { engine.run(now: true) }.keyboardShortcut("r", modifiers: .command)
            Button("Clear Text") { engine.text = "" }.keyboardShortcut(.delete, modifiers: [.command, .shift])
            Divider()
            ForEach(["I am going to the", "The weather today is", "My favourite food is"], id: \.self) { s in
                Button("Try “\(s)”") { engine.text = s + " " }
            }
            Divider()
            Button("Accept Top Suggestion") { if let s = engine.suggestions.first { engine.accept(s) } }
                .keyboardShortcut(.return, modifiers: .command)
        }
        CommandGroup(replacing: .help) {
            Button("Tip Jar…") { nav.showAbout = true }
            Link("Model Card (Hugging Face)", destination: URL(string: "https://huggingface.co/distilgpt2")!)
        }
    }
}
#endif
