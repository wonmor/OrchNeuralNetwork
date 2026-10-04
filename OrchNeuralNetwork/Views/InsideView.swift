import SwiftUI

/// Layer-by-layer visualiser: attention heatmaps, neuron activity, and the logit lens.
struct InsideView: View {
    @Environment(InferenceEngine.self) private var engine
    @State private var layer = 0
    @State private var head: Int? = nil           // nil = average of all heads
    @State private var playhead = 5
    @State private var playing = false

    var body: some View {
        @Bindable var engine = engine
        NavigationStack {
            ZStack {
                Theme.gradient.ignoresSafeArea()
                ScrollView {
                    VStack(spacing: 16) {
                        TextField("Type a sentence…", text: $engine.text, axis: .vertical)
                            .lineLimit(1...4)
                            .padding(12)
                            .background(Color.black.opacity(0.25), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                        if let trace = engine.trace {
                            stack(trace)
                            attentionCard(trace)
                            neuronsCard(trace)
                        } else {
                            Text("Type above to run the network.")
                                .foregroundStyle(.secondary).padding(.top, 40)
                        }
                    }
                    .padding()
                }
                .scrollDismissesKeyboard(.interactively)
            }
            .navigationTitle("Inside the network")
            .navigationBarTitleDisplayMode(.inline)
            .task(id: engine.trace?.id) { await replay() }
        }
    }

    // MARK: Layer stack with playback

    private func stack(_ trace: Trace) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                SectionTitle(text: "The 6 layers, in order",
                             subtitle: "Tap a layer to inspect it · bar = size of the residual vector")
                Button { Task { await replay() } } label: { Image(systemName: "play.circle.fill").font(.title2) }
                    .disabled(playing)
            }
            TokenStrip(tokens: trace.tokens, highlightLast: true)
            Text("Embedding · ‖x‖ = \(String(format: "%.1f", trace.embeddingNorm))")
                .font(.caption).foregroundStyle(.secondary)
            let maxNorm = trace.layers.map(\.residualNorm).max() ?? 1
            ForEach(trace.layers) { l in
                Button { layer = l.id } label: {
                    HStack(spacing: 10) {
                        Text("L\(l.id)").font(.system(.caption, design: .monospaced).weight(.bold)).frame(width: 26)
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                Capsule().fill(Color.white.opacity(0.06))
                                Capsule().fill(LinearGradient(colors: [Theme.accent, Theme.violet], startPoint: .leading, endPoint: .trailing))
                                    .frame(width: l.id <= playhead ? max(6, geo.size.width * CGFloat(l.residualNorm / maxNorm)) : 0)
                            }
                        }
                        .frame(height: 10)
                        VStack(alignment: .trailing, spacing: 1) {
                            if l.id <= playhead, let top = l.logitLens.first {
                                Text("→ \(top.text.trimmingCharacters(in: .whitespaces))").font(.caption.weight(.semibold)).lineLimit(1)
                                Text("\(top.prob.percent) · \(Int(l.mlpActiveFraction * 100))% neurons on")
                                    .font(.caption2).foregroundStyle(.secondary)
                            } else {
                                Text("…").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        .frame(width: 130, alignment: .trailing)
                    }
                    .padding(8)
                    .background(layer == l.id ? Theme.accent.opacity(0.18) : Color.clear,
                                in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .opacity(l.id <= playhead ? 1 : 0.35)
                }
                .buttonStyle(.plain)
            }
            Text("The arrow shows what the model would predict if it stopped after that layer (the “logit lens”). Watch the guess sharpen or change as the signal goes deeper.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .card()
        .animation(.easeOut(duration: 0.25), value: playhead)
    }

    private func replay() async {
        guard engine.trace != nil else { return }
        playing = true
        playhead = -1
        for i in 0 ..< 6 {
            try? await Task.sleep(for: .milliseconds(220))
            if Task.isCancelled { playing = false; return }
            playhead = i
        }
        playing = false
    }

    // MARK: Attention

    private func attentionCard(_ trace: Trace) -> some View {
        let l = trace.layers[min(layer, trace.layers.count - 1)]
        let T = trace.tokenCount
        let matrix = attentionMatrix(l, T: T)
        let lastRow = Array(matrix[(T - 1) * T ..< T * T])
        return VStack(alignment: .leading, spacing: 10) {
            SectionTitle(text: "Attention · layer \(l.id)",
                         subtitle: "Each row is a token looking back at earlier tokens. Brighter = more weight.")
            Picker("Head", selection: $head) {
                Text("Average").tag(Int?.none)
                ForEach(0 ..< 12, id: \.self) { h in Text("Head \(h)").tag(Int?.some(h)) }
            }
            .pickerStyle(.menu)
            .tint(Theme.accent)
            Text("What the last token “\(trace.tokens.last ?? "")” is reading from:")
                .font(.caption).foregroundStyle(.secondary)
            TokenStrip(tokens: trace.tokens, weights: lastRow, highlightLast: true)
            AttentionHeatmap(matrix: matrix, tokens: trace.tokens)
                .frame(height: min(340, CGFloat(T) * 22 + 60))
        }
        .card()
    }

    private func attentionMatrix(_ l: LayerTrace, T: Int) -> [Float] {
        if let head { return l.attention[head] }
        var avg = [Float](repeating: 0, count: T * T)
        for h in l.attention { for i in 0 ..< T * T { avg[i] += h[i] } }
        let inv = 1 / Float(l.attention.count)
        return avg.map { $0 * inv }
    }

    // MARK: Neurons

    private func neuronsCard(_ trace: Trace) -> some View {
        let l = trace.layers[min(layer, trace.layers.count - 1)]
        return VStack(alignment: .leading, spacing: 10) {
            SectionTitle(text: "MLP neurons · layer \(l.id)",
                         subtitle: "3,072 neurons for the last token · \(Int(l.mlpActiveFraction * 100))% fired")
            NeuronGrid(activations: l.mlpActivations)
                .frame(height: 120)
            Text("After attention mixes information between tokens, each token passes through this block alone. These are the raw GELU outputs: most neurons stay quiet, a few fire hard.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .card()
    }
}

struct AttentionHeatmap: View {
    let matrix: [Float]
    let tokens: [String]

    var body: some View {
        let T = tokens.count
        let image = HeatmapImage.make(values: matrix, rows: T, cols: T, mode: .sequential, scale: 1)
        GeometryReader { geo in
            let labelW: CGFloat = 54
            let side = min(geo.size.width - labelW, geo.size.height - 16)
            HStack(alignment: .top, spacing: 4) {
                VStack(spacing: 0) {
                    ForEach(Array(tokens.enumerated()), id: \.offset) { _, t in
                        Text(t.trimmingCharacters(in: .whitespaces))
                            .font(.system(size: 8, design: .monospaced)).lineLimit(1)
                            .frame(width: labelW - 4, height: side / CGFloat(T), alignment: .trailing)
                    }
                }
                if let image {
                    Image(uiImage: image)
                        .interpolation(.none)
                        .resizable()
                        .frame(width: side, height: side)
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(Theme.cardStroke))
                }
            }
        }
    }
}

struct NeuronGrid: View {
    let activations: [Float]

    var body: some View {
        let cols = 96, rows = activations.count / 96
        let clipped = activations.map { max($0, 0) }
        if let image = HeatmapImage.make(values: clipped, rows: rows, cols: cols, mode: .sequential, scale: 4) {
            Image(uiImage: image)
                .interpolation(.none)
                .resizable()
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(Theme.cardStroke))
        }
    }
}
