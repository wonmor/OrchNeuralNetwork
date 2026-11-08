import SwiftUI

/// Browse every weight matrix in the model as a heatmap.
struct WeightsView: View {
    @Environment(InferenceEngine.self) private var engine

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background
                if let file = engine.model?.file {
                    List {
                        Section {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("\(file.totalParameters.grouped) parameters").font(Theme.display(22, relativeTo: .title3))
                                Text("\(file.tensors.count) tensors  |  stored as 16-bit floats  |  \(file.header.model)")
                                    .font(.caption).foregroundStyle(Theme.muted)
                                Text("These numbers were fixed during training. Running the model only reads them.")
                                    .font(.caption).foregroundStyle(Theme.muted)
                            }
                            .listRowBackground(Theme.sheet)
                        }
                        ForEach(groups(file.tensors), id: \.title) { g in
                            Section(g.title) {
                                ForEach(g.tensors) { t in
                                    NavigationLink(value: t) {
                                        HStack {
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text(Self.pretty(t.name)).font(.body)
                                                Text(t.shapeDescription).font(.caption.monospaced()).foregroundStyle(Theme.muted)
                                            }
                                            Spacer()
                                            Text(t.count.grouped).font(.caption.monospaced()).foregroundStyle(Theme.muted)
                                        }
                                    }
                                    .listRowBackground(Theme.sheet)
                                }
                            }
                        }
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                    .navigationDestination(for: TensorInfo.self) { TensorDetailView(tensor: $0, file: file) }
                }
            }
            .navigationTitle("Weights")
            .inlineTitle()
        }
    }

    private struct Group { let title: String; let tensors: [TensorInfo] }

    private func groups(_ tensors: [TensorInfo]) -> [Group] {
        var byKey: [String: [TensorInfo]] = [:]
        var order: [String] = []
        for t in tensors {
            let key: String
            if t.name.hasPrefix("h.") {
                let n = t.name.split(separator: ".")[1]
                key = "Layer \(n)"
            } else if t.name.hasPrefix("ln_f") {
                key = "Final normalisation"
            } else {
                key = "Embeddings"
            }
            if byKey[key] == nil { order.append(key) }
            byKey[key, default: []].append(t)
        }
        return order.map { Group(title: $0, tensors: byKey[$0]!) }
    }

    static func pretty(_ name: String) -> String {
        var n = name
        if n.hasPrefix("h.") { n = String(n.split(separator: ".", maxSplits: 2)[2]) }
        let map: [String: String] = [
            "wte.weight": "Token embeddings", "wpe.weight": "Position embeddings",
            "ln_f.weight": "Final LayerNorm scale", "ln_f.bias": "Final LayerNorm shift",
            "ln_1.weight": "LayerNorm 1 scale", "ln_1.bias": "LayerNorm 1 shift",
            "attn.c_attn.weight": "Attention Q, K, V projection", "attn.c_attn.bias": "Attention Q, K, V bias",
            "attn.c_proj.weight": "Attention output projection", "attn.c_proj.bias": "Attention output bias",
            "ln_2.weight": "LayerNorm 2 scale", "ln_2.bias": "LayerNorm 2 shift",
            "mlp.c_fc.weight": "MLP expand (768 → 3072)", "mlp.c_fc.bias": "MLP expand bias",
            "mlp.c_proj.weight": "MLP contract (3072 → 768)", "mlp.c_proj.bias": "MLP contract bias",
        ]
        return map[n] ?? n
    }
}

struct TensorDetailView: View {
    let tensor: TensorInfo
    let file: WeightFile
    @State private var image: PlatformImage?
    @State private var stats: (min: Float, max: Float, mean: Float, std: Float)?
    @State private var zoom: CGFloat = 1

    var body: some View {
        ZStack {
            Theme.background
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(WeightsView.pretty(tensor.name)).font(.title3.weight(.semibold))
                        Text("\(tensor.name)  |  \(tensor.shapeDescription)  |  \(tensor.count.grouped) values")
                            .font(.caption.monospaced()).foregroundStyle(Theme.muted)
                    }
                    .card()
                    if let image {
                        ScrollView([.horizontal, .vertical]) {
                            Image(platformImage: image)
                                .interpolation(.none)
                                .resizable()
                                .aspectRatio(CGFloat(image.size.width / image.size.height), contentMode: .fit)
                                .frame(width: (PlatformLayout.isMac ? 760 : 360) * zoom)
                                .clipShape(Rectangle())
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: PlatformLayout.isMac ? 620 : 380)
                        .gesture(MagnifyGesture().onChanged { zoom = min(8, max(1, $0.magnification)) })
                        HStack {
                            Text("Zoom").font(.caption)
                            Slider(value: $zoom, in: 1 ... 8)
                        }
                        Text("Blue = negative, red = positive, pale = near zero. The grid is a sample of the full matrix; every cell is one real weight.")
                            .font(.caption).foregroundStyle(Theme.muted)
                    } else {
                        ProgressView().frame(maxWidth: .infinity, minHeight: 200)
                    }
                    if let s = stats {
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                            stat("min", s.min); stat("max", s.max); stat("mean", s.mean); stat("std", s.std)
                        }
                        .card()
                    }
                }
                .padding()
                .readableWidth()
            }
        }
        .navigationTitle("Tensor")
        .inlineTitle()
        .task(id: tensor.name) { await load() }
    }

    private func stat(_ label: String, _ v: Float) -> some View {
        HStack { Text(label).foregroundStyle(Theme.muted); Spacer(); Text(String(format: "%.4f", v)).monospaced() }
            .font(.caption)
    }

    private func load() async {
        let t = tensor, f = file
        let result: (PlatformImage?, (Float, Float, Float, Float))? = await Task.detached(priority: .userInitiated) {
            let isVector = t.shape.count == 1
            let rows = isVector ? 1 : min(t.shape[0], 256)
            let cols = isVector ? min(t.shape[0], 768) : min(t.shape.dropFirst().reduce(1, *), 256)
            guard let values = try? f.sampleGrid(t.name, rows: rows, cols: cols) else { return nil }
            let r = isVector ? 1 : rows
            let c = isVector ? values.count : cols
            let img = HeatmapImage.make(values: values, rows: r, cols: c, mode: .diverging)
            let mean = values.reduce(0, +) / Float(values.count)
            let v = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(values.count)
            return (img, (values.min() ?? 0, values.max() ?? 0, mean, v.squareRoot()))
        }.value
        if let result {
            image = result.0
            stats = (result.1.0, result.1.1, result.1.2, result.1.3)
        }
    }
}
