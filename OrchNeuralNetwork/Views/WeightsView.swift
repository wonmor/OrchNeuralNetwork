import SwiftUI

/// Browse every weight matrix in the model as a heatmap.
struct WeightsView: View {
    @Environment(InferenceEngine.self) private var engine

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.gradient.ignoresSafeArea()
                if let file = engine.model?.file {
                    List {
                        Section {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("\(file.totalParameters.grouped) parameters").font(.title3.weight(.semibold))
                                Text("\(file.tensors.count) tensors · stored as 16-bit floats · \(file.header.model)")
                                    .font(.caption).foregroundStyle(.secondary)
                                Text("These numbers were fixed during training. Running the model only reads them.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            .listRowBackground(Theme.card)
                        }
                        ForEach(groups(file.tensors), id: \.title) { g in
                            Section(g.title) {
                                ForEach(g.tensors) { t in
                                    NavigationLink(value: t) {
                                        HStack {
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text(Self.pretty(t.name)).font(.body)
                                                Text(t.shapeDescription).font(.caption.monospaced()).foregroundStyle(.secondary)
                                            }
                                            Spacer()
                                            Text(t.count.grouped).font(.caption.monospaced()).foregroundStyle(.secondary)
                                        }
                                    }
                                    .listRowBackground(Theme.card)
                                }
                            }
                        }
                    }
                    .scrollContentBackground(.hidden)
                    .navigationDestination(for: TensorInfo.self) { TensorDetailView(tensor: $0, file: file) }
                }
            }
            .navigationTitle("Weights")
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
            "attn.c_attn.weight": "Attention Q·K·V projection", "attn.c_attn.bias": "Attention Q·K·V bias",
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
    @State private var image: UIImage?
    @State private var stats: (min: Float, max: Float, mean: Float, std: Float)?
    @State private var zoom: CGFloat = 1

    var body: some View {
        ZStack {
            Theme.gradient.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(WeightsView.pretty(tensor.name)).font(.title3.weight(.semibold))
                        Text("\(tensor.name) · \(tensor.shapeDescription) · \(tensor.count.grouped) values")
                            .font(.caption.monospaced()).foregroundStyle(.secondary)
                    }
                    .card()
                    if let image {
                        ScrollView([.horizontal, .vertical]) {
                            Image(uiImage: image)
                                .interpolation(.none)
                                .resizable()
                                .aspectRatio(CGFloat(image.size.width / image.size.height), contentMode: .fit)
                                .frame(width: 360 * zoom)
                                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 380)
                        .gesture(MagnifyGesture().onChanged { zoom = min(8, max(1, $0.magnification)) })
                        HStack {
                            Text("Zoom").font(.caption)
                            Slider(value: $zoom, in: 1 ... 8)
                        }
                        Text("Blue = negative, orange = positive, dark = near zero. The grid is a sample of the full matrix; every cell is one real weight.")
                            .font(.caption).foregroundStyle(.secondary)
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
            }
        }
        .navigationTitle("Tensor")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: tensor.name) { await load() }
    }

    private func stat(_ label: String, _ v: Float) -> some View {
        HStack { Text(label).foregroundStyle(.secondary); Spacer(); Text(String(format: "%.4f", v)).monospaced() }
            .font(.caption)
    }

    private func load() async {
        let t = tensor, f = file
        let result: (UIImage?, (Float, Float, Float, Float))? = await Task.detached(priority: .userInitiated) {
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
