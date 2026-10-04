import Foundation
import Accelerate

/// Describes one tensor inside a `.tsw` weight file.
struct TensorInfo: Decodable, Identifiable, Hashable {
    let name: String
    let shape: [Int]
    let dtype: String
    let offset: Int
    let nbytes: Int

    var id: String { name }
    var count: Int { shape.reduce(1, *) }
    var shapeDescription: String { shape.map(String.init).joined(separator: " × ") }
}

struct ModelConfig: Decodable {
    let n_layer: Int
    let n_head: Int
    let n_embd: Int
    let n_ctx: Int
    let n_vocab: Int
    let layer_norm_epsilon: Float
    var headDim: Int { n_embd / n_head }
}

struct WeightHeader: Decodable {
    let model: String
    let license: String?
    let source: String?
    let config: ModelConfig
    let tensors: [TensorInfo]
}

enum WeightFileError: Error, LocalizedError {
    case badMagic, truncated, missingTensor(String), badDType(String)
    var errorDescription: String? {
        switch self {
        case .badMagic: return "The weight file has an unexpected format."
        case .truncated: return "The weight file is incomplete."
        case .missingTensor(let n): return "Tensor \(n) is missing from the weight file."
        case .badDType(let d): return "Unsupported tensor type \(d)."
        }
    }
}

/// Memory-mapped reader for the app's compact fp16 weight container.
///
/// Layout: "TSW1" magic, UInt32 LE header length, UTF-8 JSON header, zero padding
/// to a 64-byte boundary, then raw little-endian float16 tensors at 64-byte-aligned offsets.
final class WeightFile {
    let header: WeightHeader
    private let data: Data
    private let dataStart: Int
    private let index: [String: TensorInfo]

    var config: ModelConfig { header.config }
    var tensors: [TensorInfo] { header.tensors }
    var totalParameters: Int { header.tensors.reduce(0) { $0 + $1.count } }

    init(url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 8, data.prefix(4) == Data("TSW1".utf8) else { throw WeightFileError.badMagic }
        let headerLength = data.withUnsafeBytes { raw -> Int in
            Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self)))
        }
        guard data.count >= 8 + headerLength else { throw WeightFileError.truncated }
        let header = try JSONDecoder().decode(WeightHeader.self, from: data.subdata(in: 8 ..< 8 + headerLength))
        self.data = data
        self.header = header
        self.dataStart = (8 + headerLength + 63) / 64 * 64
        self.index = Dictionary(uniqueKeysWithValues: header.tensors.map { ($0.name, $0) })
        for t in header.tensors where dataStart + t.offset + t.nbytes > data.count {
            throw WeightFileError.truncated
        }
    }

    func info(_ name: String) throws -> TensorInfo {
        guard let t = index[name] else { throw WeightFileError.missingTensor(name) }
        return t
    }

    /// Loads a tensor as Float32, converting from float16 with vImage.
    func floats(_ name: String) throws -> [Float] {
        let t = try info(name)
        guard t.dtype == "f16" else { throw WeightFileError.badDType(t.dtype) }
        let count = t.count
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            let src = raw.baseAddress!.advanced(by: dataStart + t.offset)
            out.withUnsafeMutableBufferPointer { dst in
                var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1,
                                           width: vImagePixelCount(count), rowBytes: count * 2)
                var dstBuf = vImage_Buffer(data: dst.baseAddress!, height: 1,
                                           width: vImagePixelCount(count), rowBytes: count * 4)
                vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
        }
        return out
    }

    /// Samples a 2-D tensor on a coarse grid without materialising the whole thing as Float32.
    /// Returns row-major `rows × cols` values.
    func sampleGrid(_ name: String, rows: Int, cols: Int) throws -> [Float] {
        let t = try info(name)
        let (h, w): (Int, Int) = t.shape.count >= 2 ? (t.shape[0], t.shape.dropFirst().reduce(1, *)) : (1, t.shape[0])
        let rs = max(1, min(rows, h)), cs = max(1, min(cols, w))
        var out = [Float](repeating: 0, count: rs * cs)
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: dataStart + t.offset).assumingMemoryBound(to: UInt16.self)
            for r in 0 ..< rs {
                let srcRow = r * h / rs
                for c in 0 ..< cs {
                    let srcCol = c * w / cs
                    out[r * cs + c] = Float(Float16(bitPattern: base[srcRow * w + srcCol]))
                }
            }
        }
        return out
    }
}
