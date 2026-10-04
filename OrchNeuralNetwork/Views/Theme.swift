import SwiftUI
import UIKit

enum Theme {
    static let background = Color(red: 0.031, green: 0.055, blue: 0.145)
    static let backgroundTop = Color(red: 0.07, green: 0.09, blue: 0.24)
    static let accent = Color(red: 0.38, green: 0.69, blue: 1.0)
    static let violet = Color(red: 0.64, green: 0.47, blue: 1.0)
    static let mint = Color(red: 0.42, green: 0.93, blue: 0.78)
    static let amber = Color(red: 1.0, green: 0.72, blue: 0.30)
    static let card = Color.white.opacity(0.07)
    static let cardStroke = Color.white.opacity(0.10)

    static var gradient: LinearGradient {
        LinearGradient(colors: [backgroundTop, background], startPoint: .top, endPoint: .bottom)
    }

    /// Sequential colour ramp used for attention and activations: navy → blue → cyan → white.
    static func heat(_ v: Float) -> Color {
        let t = Double(min(max(v, 0), 1))
        if t < 0.5 {
            let k = t / 0.5
            return Color(red: 0.06 + 0.22 * k, green: 0.08 + 0.52 * k, blue: 0.25 + 0.75 * k)
        } else {
            let k = (t - 0.5) / 0.5
            return Color(red: 0.28 + 0.72 * k, green: 0.60 + 0.40 * k, blue: 1.0)
        }
    }
}

struct CardModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(14)
            .background(Theme.card, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.cardStroke, lineWidth: 1))
    }
}

extension View {
    func card() -> some View { modifier(CardModifier()) }
}

struct SectionTitle: View {
    let text: String
    var subtitle: String? = nil
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(text).font(.headline)
            if let subtitle { Text(subtitle).font(.caption).foregroundStyle(.secondary) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Builds small raster heatmaps from float arrays. Images are far cheaper than thousands of SwiftUI rects.
enum HeatmapImage {
    enum Mode { case sequential, diverging }

    static func make(values: [Float], rows: Int, cols: Int, mode: Mode, scale: Float? = nil) -> UIImage? {
        guard rows > 0, cols > 0, values.count >= rows * cols else { return nil }
        var pixels = [UInt8](repeating: 255, count: rows * cols * 4)
        let s: Float
        if let scale { s = scale } else {
            switch mode {
            case .sequential: s = max(values.max() ?? 1, 1e-6)
            case .diverging:
                let mean = values.reduce(0, +) / Float(values.count)
                let varSum = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
                s = max(2.5 * (varSum / Float(values.count)).squareRoot(), 1e-6)
            }
        }
        for i in 0 ..< rows * cols {
            let v = values[i] / s
            var r: Float, g: Float, b: Float
            switch mode {
            case .sequential:
                let t = min(max(v, 0), 1)
                if t < 0.5 { let k = t / 0.5; r = 0.06 + 0.22 * k; g = 0.08 + 0.52 * k; b = 0.25 + 0.75 * k }
                else { let k = (t - 0.5) / 0.5; r = 0.28 + 0.72 * k; g = 0.60 + 0.40 * k; b = 1 }
            case .diverging:
                let t = min(max(v, -1), 1)
                if t < 0 { let k = -t; r = 0.05 + 0.10 * k; g = 0.07 + 0.45 * k; b = 0.15 + 0.85 * k }
                else { let k = t; r = 0.05 + 0.95 * k; g = 0.07 + 0.55 * k; b = 0.15 + 0.10 * k }
            }
            pixels[i * 4] = UInt8(max(0, min(255, r * 255)))
            pixels[i * 4 + 1] = UInt8(max(0, min(255, g * 255)))
            pixels[i * 4 + 2] = UInt8(max(0, min(255, b * 255)))
        }
        let data = Data(pixels)
        guard let provider = CGDataProvider(data: data as CFData),
              let cg = CGImage(width: cols, height: rows, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: cols * 4,
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { return nil }
        return UIImage(cgImage: cg)
    }
}

struct TokenChip: View {
    let text: String
    var weight: Float = 0          // 0…1 highlight intensity
    var highlighted = false

    var body: some View {
        Text(display)
            .font(.system(.footnote, design: .monospaced))
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(weight > 0.02 ? Theme.heat(weight).opacity(0.85) : Color.white.opacity(0.08),
                        in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous)
                .stroke(highlighted ? Theme.amber : Color.white.opacity(0.12), lineWidth: highlighted ? 2 : 1))
            .foregroundStyle(weight > 0.6 ? Color.black : Color.white)
    }

    private var display: String {
        let s = text.replacingOccurrences(of: "\n", with: "⏎")
        return s.hasPrefix(" ") ? "␣" + s.dropFirst() : (s.isEmpty ? "∅" : s)
    }
}

struct TokenStrip: View {
    let tokens: [String]
    var weights: [Float]? = nil
    var highlightLast = false

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Array(tokens.enumerated()), id: \.offset) { i, t in
                    TokenChip(text: t, weight: weights?[i] ?? 0, highlighted: highlightLast && i == tokens.count - 1)
                }
            }
            .padding(.vertical, 2)
        }
    }
}

struct ProbabilityBars: View {
    let items: [TokenProb]
    var limit = 10

    var body: some View {
        let shown = Array(items.prefix(limit))
        let maxP = max(shown.first?.prob ?? 1, 1e-6)
        VStack(spacing: 6) {
            ForEach(shown) { item in
                HStack(spacing: 8) {
                    TokenChip(text: item.text).frame(width: 96, alignment: .leading)
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color.white.opacity(0.06))
                            Capsule().fill(LinearGradient(colors: [Theme.accent, Theme.violet], startPoint: .leading, endPoint: .trailing))
                                .frame(width: max(4, geo.size.width * CGFloat(item.prob / maxP)))
                        }
                    }
                    .frame(height: 12)
                    Text(item.prob.percent).font(.system(.caption, design: .monospaced)).frame(width: 52, alignment: .trailing)
                        .foregroundStyle(.secondary)
                }
                .animation(.easeOut(duration: 0.3), value: item.prob)
            }
        }
    }
}

extension Float {
    var percent: String {
        self >= 0.1 ? String(format: "%.0f%%", self * 100)
            : self >= 0.01 ? String(format: "%.1f%%", self * 100)
            : String(format: "%.2f%%", self * 100)
    }
}

extension Int {
    var grouped: String { NumberFormatter.grouped.string(from: NSNumber(value: self)) ?? String(self) }
}

extension NumberFormatter {
    static let grouped: NumberFormatter = { let f = NumberFormatter(); f.numberStyle = .decimal; return f }()
}
