import SwiftUI

/// The "keyboard" demo: type, get QuickType-style suggestions, and watch the network fire.
struct PredictView: View {
    @Environment(InferenceEngine.self) private var engine
    @State private var showAbout = false
    @FocusState private var focused: Bool

    var body: some View {
        @Bindable var engine = engine
        NavigationStack {
            ZStack {
                Theme.gradient.ignoresSafeArea()
                ScrollView {
                    VStack(spacing: 16) {
                        inputCard(text: $engine.text)
                        suggestionBar
                        pulseCard
                        if let trace = engine.trace {
                            VStack(alignment: .leading, spacing: 10) {
                                SectionTitle(text: "What the model thinks comes next",
                                             subtitle: "Probability over all 50,257 tokens, top 10 shown")
                                ProbabilityBars(items: trace.topK)
                            }
                            .card()
                            VStack(alignment: .leading, spacing: 10) {
                                SectionTitle(text: "Your text as tokens", subtitle: "\(trace.tokenCount) tokens · ␣ marks a leading space")
                                TokenStrip(tokens: trace.tokens, highlightLast: true)
                            }
                            .card()
                        }
                    }
                    .padding()
                }
                .scrollDismissesKeyboard(.interactively)
            }
            .navigationTitle("Orch Neural Network")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showAbout = true } label: { Image(systemName: "info.circle") }
                }
                ToolbarItem(placement: .keyboard) {
                    HStack {
                        Spacer()
                        Button("Done") { focused = false }
                    }
                }
            }
            .sheet(isPresented: $showAbout) { AboutView() }
        }
    }

    private func inputCard(text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                SectionTitle(text: "Type something", subtitle: "A 6-layer GPT-2 predicts the next word live")
                if self.engine.isRunning { ProgressView().controlSize(.small) }
            }
            TextField("I am going to the", text: text, axis: .vertical)
                .focused($focused)
                .lineLimit(2...6)
                .font(.title3)
                .padding(12)
                .background(Color.black.opacity(0.25), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
            HStack {
                ForEach(["I am going to the", "The weather today is", "My favourite food is"], id: \.self) { s in
                    Button(s.split(separator: " ").prefix(2).joined(separator: " ") + "…") { self.engine.text = s + " " }
                        .font(.caption2)
                        .buttonStyle(.bordered)
                        .tint(Theme.violet)
                }
            }
            .lineLimit(1)
        }
        .card()
    }

    private var suggestionBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionTitle(text: "Suggestions", subtitle: "Exactly what a keyboard shows: the three most probable next words")
            HStack(spacing: 8) {
                if engine.suggestions.isEmpty {
                    Text(engine.text.isEmpty ? "Start typing…" : "No word-like suggestions")
                        .font(.footnote).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, minHeight: 44)
                } else {
                    ForEach(engine.suggestions) { s in
                        Button {
                            engine.accept(s)
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        } label: {
                            VStack(spacing: 2) {
                                Text(s.display).font(.body.weight(.medium)).lineLimit(1).minimumScaleFactor(0.7)
                                Text(s.prob.percent).font(.caption2).foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, minHeight: 44)
                        }
                        .buttonStyle(.plain)
                        .padding(.vertical, 4)
                        .background(Color.white.opacity(0.10), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }
                }
            }
        }
        .card()
    }

    private var pulseCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                SectionTitle(text: "Signal through the network",
                             subtitle: engine.trace.map { "6 layers · \(Int($0.elapsedMilliseconds)) ms on device" } ?? "Waiting for input")
                NavigationLink { InsideView() } label: { Text("Open visualiser").font(.caption) }
            }
            LayerPulseView(trace: engine.trace)
        }
        .card()
    }
}

/// Six bars that light up in sequence every time a forward pass completes.
struct LayerPulseView: View {
    let trace: Trace?
    @State private var lit = -1

    var body: some View {
        HStack(spacing: 6) {
            ForEach(0 ..< 6, id: \.self) { i in
                VStack(spacing: 4) {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(i <= lit ? AnyShapeStyle(LinearGradient(colors: [Theme.accent, Theme.violet], startPoint: .bottom, endPoint: .top))
                                       : AnyShapeStyle(Color.white.opacity(0.08)))
                        .frame(height: 28 + CGFloat(normNorm(i)) * 26)
                        .shadow(color: i == lit ? Theme.accent.opacity(0.8) : .clear, radius: 8)
                    Text(i < 6 ? "L\(i)" : "").font(.caption2).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
            }
        }
        .frame(height: 70, alignment: .bottom)
        .animation(.easeOut(duration: 0.12), value: lit)
        .task(id: trace?.id) {
            guard trace != nil else { lit = -1; return }
            lit = -1
            for i in 0 ..< 6 {
                try? await Task.sleep(for: .milliseconds(90))
                if Task.isCancelled { return }
                lit = i
            }
        }
    }

    private func normNorm(_ i: Int) -> Float {
        guard let trace, i < trace.layers.count else { return 0 }
        let maxN = trace.layers.map(\.residualNorm).max() ?? 1
        return trace.layers[i].residualNorm / max(maxN, 1e-6)
    }
}
