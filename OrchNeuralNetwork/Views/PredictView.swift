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
                Theme.background
                ScrollView {
                    VStack(spacing: 16) {
                        BauhausBanner()
                        inputCard(text: $engine.text)
                        suggestionBar
                        pulseCard
                        if let trace = engine.trace {
                            VStack(alignment: .leading, spacing: 10) {
                                SectionTitle(text: "What the model thinks comes next",
                                             subtitle: "Probability over all 50,257 tokens, top 10 shown", mark: .triangle, color: Theme.yellow)
                                ProbabilityBars(items: trace.topK)
                            }
                            .card()
                            VStack(alignment: .leading, spacing: 10) {
                                SectionTitle(text: "Your text as tokens", subtitle: "\(trace.tokenCount) tokens · ␣ marks a leading space", mark: .circle, color: Theme.red)
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
                SectionTitle(text: "Type something", subtitle: "A 6-layer GPT-2 predicts the next word live", mark: .circle, color: Theme.blue)
                if self.engine.isRunning { ProgressView().controlSize(.small) }
            }
            TextField("I am going to the", text: text, axis: .vertical)
                .focused($focused)
                .lineLimit(2...6)
                .font(.title3)
                .padding(12)
                .background(Theme.sheet)
                .overlay(Rectangle().stroke(Theme.ink, lineWidth: Theme.rule))
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
            HStack {
                ForEach(["I am going to the", "The weather today is", "My favourite food is"], id: \.self) { s in
                    Button { self.engine.text = s + " " } label: {
                        Text(s.split(separator: " ").prefix(2).joined(separator: " ") + "…")
                            .font(.caption.weight(.semibold)).padding(.horizontal, 10).padding(.vertical, 6)
                    }
                    .buttonStyle(BlockButtonStyle())
                }
            }
            .lineLimit(1)
        }
        .card()
    }

    private var suggestionBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionTitle(text: "Suggestions", subtitle: "Exactly what a keyboard shows: the three most probable next words", mark: .triangle, color: Theme.red)
            HStack(spacing: 8) {
                if engine.suggestions.isEmpty {
                    Text(engine.text.isEmpty ? "Start typing…" : "No word-like suggestions")
                        .font(.footnote).foregroundStyle(Theme.muted)
                        .frame(maxWidth: .infinity, minHeight: 44)
                } else {
                    ForEach(Array(engine.suggestions.enumerated()), id: \.element.id) { idx, s in
                        Button {
                            engine.accept(s)
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        } label: {
                            VStack(spacing: 2) {
                                Text(s.display).font(.body.weight(.medium)).lineLimit(1).minimumScaleFactor(0.7)
                                Text(s.prob.percent).font(.caption2).foregroundStyle(Theme.muted)
                            }
                            .frame(maxWidth: .infinity, minHeight: 44)
                        }
                        .padding(.vertical, 4)
                        .buttonStyle(BlockButtonStyle(fill: idx == 0 ? Theme.yellow : Theme.sheet))
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
                             subtitle: engine.trace.map { "6 layers · \(Int($0.elapsedMilliseconds)) ms on device" } ?? "Waiting for input", mark: .square, color: Theme.blue)
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
                    Rectangle()
                        .fill(i <= lit ? AnyShapeStyle(i == lit ? Theme.red : Theme.blue)
                                       : AnyShapeStyle(Theme.ink.opacity(0.08)))
                        .frame(height: 28 + CGFloat(normNorm(i)) * 26)
                    Text(i < 6 ? "L\(i)" : "").font(.caption2).foregroundStyle(Theme.muted)
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

/// Header composition: input tokens (circles) feed layers (squares) that produce a prediction (triangle).
struct BauhausBanner: View {
    var body: some View {
        HStack(spacing: 14) {
            Canvas { ctx, size in
                let w = size.width, h = size.height
                let ins = [CGPoint(x: w * 0.12, y: h * 0.2), CGPoint(x: w * 0.12, y: h * 0.5), CGPoint(x: w * 0.12, y: h * 0.8)]
                let mid = [CGPoint(x: w * 0.5, y: h * 0.32), CGPoint(x: w * 0.5, y: h * 0.7)]
                let out = CGPoint(x: w * 0.86, y: h * 0.5)
                ctx.fill(Path(ellipseIn: CGRect(x: w * 0.3, y: -h * 0.05, width: h * 1.1, height: h * 1.1)), with: .color(Theme.yellow))
                var lines = Path()
                for a in ins { for b in mid { lines.move(to: a); lines.addLine(to: b) } }
                for a in mid { lines.move(to: a); lines.addLine(to: out) }
                ctx.stroke(lines, with: .color(Theme.ink), lineWidth: 3)
                for p in ins { ctx.fill(Path(ellipseIn: CGRect(x: p.x - 7, y: p.y - 7, width: 14, height: 14)), with: .color(Theme.ink)) }
                ctx.fill(Path(ellipseIn: CGRect(x: mid[0].x - 11, y: mid[0].y - 11, width: 22, height: 22)), with: .color(Theme.red))
                ctx.fill(Path(CGRect(x: mid[1].x - 10, y: mid[1].y - 10, width: 20, height: 20)), with: .color(Theme.blue))
                ctx.fill(Triangle().path(in: CGRect(x: out.x - 12, y: out.y - 14, width: 24, height: 28)), with: .color(Theme.ink))
            }
            .frame(width: 120, height: 76)
            VStack(alignment: .leading, spacing: 4) {
                Text("WATCH A LANGUAGE MODEL THINK").font(Theme.display(17, relativeTo: .title3)).lineLimit(2).minimumScaleFactor(0.8)
                Text("82 million weights · 6 layers · on this device").font(.caption).foregroundStyle(Theme.muted)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .padding(.leading, 6)
        .background(Theme.sheet)
        .overlay(alignment: .leading) { Rectangle().fill(Theme.red).frame(width: 8) }
        .overlay(Rectangle().stroke(Theme.ink, lineWidth: Theme.rule))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Watch a language model think. 82 million weights, 6 layers, on this device.")
    }
}
