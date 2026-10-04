import Foundation
import Accelerate

struct TokenProb: Identifiable, Hashable {
    let id: Int
    let text: String
    let prob: Float
}

/// Everything recorded while one forward pass ran, for visualisation.
struct LayerTrace: Identifiable {
    let id: Int
    /// attention[head] is a T×T row-major matrix: row = query position, column = key position.
    let attention: [[Float]]
    /// Top predictions if the model stopped after this layer ("logit lens").
    let logitLens: [TokenProb]
    /// L2 norm of the last token's residual stream after this layer.
    let residualNorm: Float
    /// GELU outputs of the last token's 3072 MLP neurons.
    let mlpActivations: [Float]
    /// Fraction of MLP neurons with activation > 0.
    let mlpActiveFraction: Float
}

struct Trace: Identifiable {
    let id = UUID()
    let tokenIDs: [Int]
    let tokens: [String]
    let embeddingNorm: Float
    /// First 64 values of the last token's embedding (for display).
    let embeddingPreview: [Float]
    let layers: [LayerTrace]
    let topK: [TokenProb]
    let elapsedMilliseconds: Double
    var tokenCount: Int { tokens.count }
}

/// A from-scratch GPT-2 implementation on Accelerate. Single-threaded, no KV cache: every call
/// recomputes the whole sequence so that every intermediate value can be captured.
final class GPT2Model {
    let config: ModelConfig
    let file: WeightFile
    let tokenizer: BPETokenizer

    private let wte: [Float], wpe: [Float], lnfG: [Float], lnfB: [Float]
    private struct Block {
        let ln1g: [Float], ln1b: [Float]
        let attnW: [Float], attnB: [Float]
        let projW: [Float], projB: [Float]
        let ln2g: [Float], ln2b: [Float]
        let fcW: [Float], fcB: [Float]
        let mlpProjW: [Float], mlpProjB: [Float]
    }
    private let blocks: [Block]

    init(file: WeightFile, tokenizer: BPETokenizer) throws {
        self.file = file
        self.tokenizer = tokenizer
        self.config = file.config
        wte = try file.floats("wte.weight")
        wpe = try file.floats("wpe.weight")
        lnfG = try file.floats("ln_f.weight")
        lnfB = try file.floats("ln_f.bias")
        var blocks: [Block] = []
        for i in 0 ..< config.n_layer {
            let p = "h.\(i)."
            blocks.append(Block(
                ln1g: try file.floats(p + "ln_1.weight"), ln1b: try file.floats(p + "ln_1.bias"),
                attnW: try file.floats(p + "attn.c_attn.weight"), attnB: try file.floats(p + "attn.c_attn.bias"),
                projW: try file.floats(p + "attn.c_proj.weight"), projB: try file.floats(p + "attn.c_proj.bias"),
                ln2g: try file.floats(p + "ln_2.weight"), ln2b: try file.floats(p + "ln_2.bias"),
                fcW: try file.floats(p + "mlp.c_fc.weight"), fcB: try file.floats(p + "mlp.c_fc.bias"),
                mlpProjW: try file.floats(p + "mlp.c_proj.weight"), mlpProjB: try file.floats(p + "mlp.c_proj.bias")))
        }
        self.blocks = blocks
    }

    static func loadBundled() throws -> GPT2Model {
        let bundle = Bundle.main
        guard let w = bundle.url(forResource: "distilgpt2", withExtension: "tsw"),
              let v = bundle.url(forResource: "vocab", withExtension: "json"),
              let m = bundle.url(forResource: "merges", withExtension: "txt") else {
            throw WeightFileError.missingTensor("bundle resources")
        }
        let file = try WeightFile(url: w)
        let tok = try BPETokenizer(vocabURL: v, mergesURL: m)
        return try GPT2Model(file: file, tokenizer: tok)
    }

    // MARK: Forward pass

    func run(text: String, maxTokens: Int = 48, topK: Int = 12) -> Trace? {
        // A keyboard predicts the *next word* after a space, so trailing whitespace is dropped before
        // tokenizing; otherwise GPT-2 turns the lone space into its own token and predicts word fragments.
        var cleaned = text
        while let last = cleaned.last, last.isWhitespace { cleaned.removeLast() }
        var ids = tokenizer.encode(cleaned)
        if ids.isEmpty { return nil }
        if ids.count > maxTokens { ids = Array(ids.suffix(maxTokens)) }
        return run(ids: ids, topK: topK)
    }

    func run(ids: [Int], topK: Int = 12) -> Trace {
        let start = CFAbsoluteTimeGetCurrent()
        let T = ids.count, D = config.n_embd, H = config.n_head, hd = config.headDim
        let F = 4 * D, V = config.n_vocab

        // x[T, D] = wte[id] + wpe[pos]
        var x = [Float](repeating: 0, count: T * D)
        for t in 0 ..< T {
            let id = ids[t]
            vDSP_vadd(Array(wte[id * D ..< (id + 1) * D]), 1, Array(wpe[t * D ..< (t + 1) * D]), 1, &x[t * D], 1, vDSP_Length(D))
        }
        let embeddingNorm = norm(x, offset: (T - 1) * D, count: D)
        let embeddingPreview = Array(x[(T - 1) * D ..< (T - 1) * D + 64])

        var layerTraces: [LayerTrace] = []
        var h = [Float](repeating: 0, count: T * D)
        var qkv = [Float](repeating: 0, count: T * 3 * D)
        var attnOut = [Float](repeating: 0, count: T * D)
        var scores = [Float](repeating: 0, count: T * T)
        var fc = [Float](repeating: 0, count: T * F)
        var tmp = [Float](repeating: 0, count: T * D)

        for (li, b) in blocks.enumerated() {
            // --- attention ---
            layerNorm(x, into: &h, T: T, D: D, g: b.ln1g, beta: b.ln1b)
            matmul(h, T, D, b.attnW, 3 * D, into: &qkv)
            addBias(&qkv, b.attnB, rows: T, cols: 3 * D)

            var heads: [[Float]] = []
            heads.reserveCapacity(H)
            let scale = 1 / Float(hd).squareRoot()
            qkv.withUnsafeBufferPointer { q in
                attnOut.withUnsafeMutableBufferPointer { o in
                    for head in 0 ..< H {
                        let qp = q.baseAddress! + head * hd
                        let kp = q.baseAddress! + D + head * hd
                        let vp = q.baseAddress! + 2 * D + head * hd
                        scores.withUnsafeMutableBufferPointer { s in
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(T), Int32(T), Int32(hd),
                                        scale, qp, Int32(3 * D), kp, Int32(3 * D), 0, s.baseAddress!, Int32(T))
                        }
                        causalSoftmax(&scores, T: T)
                        heads.append(scores)
                        scores.withUnsafeBufferPointer { s in
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(T), Int32(hd), Int32(T),
                                        1, s.baseAddress!, Int32(T), vp, Int32(3 * D), 0, o.baseAddress! + head * hd, Int32(D))
                        }
                    }
                }
            }
            matmul(attnOut, T, D, b.projW, D, into: &tmp)
            addBias(&tmp, b.projB, rows: T, cols: D)
            vDSP_vadd(x, 1, tmp, 1, &x, 1, vDSP_Length(T * D))

            // --- MLP ---
            layerNorm(x, into: &h, T: T, D: D, g: b.ln2g, beta: b.ln2b)
            matmul(h, T, D, b.fcW, F, into: &fc)
            addBias(&fc, b.fcB, rows: T, cols: F)
            gelu(&fc)
            let mlpAct = Array(fc[(T - 1) * F ..< T * F])
            matmul(fc, T, F, b.mlpProjW, D, into: &tmp)
            addBias(&tmp, b.mlpProjB, rows: T, cols: D)
            vDSP_vadd(x, 1, tmp, 1, &x, 1, vDSP_Length(T * D))

            // --- logit lens on the last token ---
            let lens = predict(from: x, T: T, D: D, V: V, topK: 5)
            let active = Float(mlpAct.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }) / Float(F)
            layerTraces.append(LayerTrace(id: li, attention: heads, logitLens: lens,
                                          residualNorm: norm(x, offset: (T - 1) * D, count: D),
                                          mlpActivations: mlpAct, mlpActiveFraction: active))
        }

        let top = predict(from: x, T: T, D: D, V: V, topK: topK)
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
        return Trace(tokenIDs: ids, tokens: ids.map { tokenizer.decode($0) },
                     embeddingNorm: embeddingNorm, embeddingPreview: embeddingPreview,
                     layers: layerTraces, topK: top, elapsedMilliseconds: ms)
    }

    // MARK: Kernels

    private func predict(from x: [Float], T: Int, D: Int, V: Int, topK: Int) -> [TokenProb] {
        var last = Array(x[(T - 1) * D ..< T * D])
        var normed = [Float](repeating: 0, count: D)
        layerNorm(last, into: &normed, T: 1, D: D, g: lnfG, beta: lnfB)
        last = normed
        var logits = [Float](repeating: 0, count: V)
        normed.withUnsafeBufferPointer { n in
            wte.withUnsafeBufferPointer { w in
                logits.withUnsafeMutableBufferPointer { l in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, Int32(V), Int32(D),
                                1, n.baseAddress!, Int32(D), w.baseAddress!, Int32(D), 0, l.baseAddress!, Int32(V))
                }
            }
        }
        var maxv: Float = 0
        vDSP_maxv(logits, 1, &maxv, vDSP_Length(V))
        var neg = -maxv
        vDSP_vsadd(logits, 1, &neg, &logits, 1, vDSP_Length(V))
        var n32 = Int32(V)
        vvexpf(&logits, logits, &n32)
        var sum: Float = 0
        vDSP_sve(logits, 1, &sum, vDSP_Length(V))
        var inv = 1 / sum
        vDSP_vsmul(logits, 1, &inv, &logits, 1, vDSP_Length(V))
        // partial top-k
        var best: [(Int, Float)] = []
        best.reserveCapacity(topK + 1)
        for i in 0 ..< V {
            let p = logits[i]
            if best.count < topK || p > best[best.count - 1].1 {
                var j = best.count
                best.append((i, p))
                while j > 0, best[j - 1].1 < p { best[j] = best[j - 1]; j -= 1 }
                best[j] = (i, p)
                if best.count > topK { best.removeLast() }
            }
        }
        return best.map { TokenProb(id: $0.0, text: tokenizer.decode($0.0), prob: $0.1) }
    }

    private func matmul(_ a: [Float], _ m: Int, _ k: Int, _ b: [Float], _ n: Int, into c: inout [Float]) {
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k),
                                1, ap.baseAddress!, Int32(k), bp.baseAddress!, Int32(n), 0, cp.baseAddress!, Int32(n))
                }
            }
        }
    }

    private func addBias(_ y: inout [Float], _ bias: [Float], rows: Int, cols: Int) {
        for r in 0 ..< rows {
            vDSP_vadd(Array(y[r * cols ..< (r + 1) * cols]), 1, bias, 1, &y[r * cols], 1, vDSP_Length(cols))
        }
    }

    private func layerNorm(_ x: [Float], into y: inout [Float], T: Int, D: Int, g: [Float], beta: [Float]) {
        let eps = config.layer_norm_epsilon
        x.withUnsafeBufferPointer { xp in
            y.withUnsafeMutableBufferPointer { yp in
                for t in 0 ..< T {
                    let row = xp.baseAddress! + t * D
                    var mean: Float = 0, meanSq: Float = 0
                    vDSP_meanv(row, 1, &mean, vDSP_Length(D))
                    vDSP_measqv(row, 1, &meanSq, vDSP_Length(D))
                    let variance = max(0, meanSq - mean * mean)   // population variance, as GPT-2 uses
                    let inv = 1 / (variance + eps).squareRoot()
                    let out = yp.baseAddress! + t * D
                    for i in 0 ..< D {
                        out[i] = (row[i] - mean) * inv * g[i] + beta[i]
                    }
                }
            }
        }
    }

    private func causalSoftmax(_ s: inout [Float], T: Int) {
        for i in 0 ..< T {
            let row = i * T
            var m = -Float.greatestFiniteMagnitude
            for j in 0 ... i { m = max(m, s[row + j]) }
            var sum: Float = 0
            for j in 0 ... i { let e = expf(s[row + j] - m); s[row + j] = e; sum += e }
            let inv = 1 / sum
            for j in 0 ... i { s[row + j] *= inv }
            if i + 1 < T { for j in (i + 1) ..< T { s[row + j] = 0 } }
        }
    }

    private func gelu(_ v: inout [Float]) {
        let c: Float = 0.7978845608 // sqrt(2/pi)
        for i in 0 ..< v.count {
            let x = v[i]
            v[i] = 0.5 * x * (1 + tanhf(c * (x + 0.044715 * x * x * x)))
        }
    }

    private func norm(_ v: [Float], offset: Int, count: Int) -> Float {
        var r: Float = 0
        v.withUnsafeBufferPointer { p in vDSP_svesq(p.baseAddress! + offset, 1, &r, vDSP_Length(count)) }
        return r.squareRoot()
    }
}
