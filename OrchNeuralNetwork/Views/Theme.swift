import SwiftUI
import UIKit

/// Bauhaus palette: three primaries and black on warm paper. Shapes carry meaning:
/// circle = token / neuron, square = layer / matrix, triangle = prediction.
enum Theme {
    static let paper = Color(red: 0.949, green: 0.925, blue: 0.878)
    static let sheet = Color(red: 0.988, green: 0.976, blue: 0.949)
    static let ink = Color(red: 0.078, green: 0.078, blue: 0.078)
    static let red = Color(red: 0.835, green: 0.169, blue: 0.118)
    static let yellow = Color(red: 0.957, green: 0.745, blue: 0.0)
    static let blue = Color(red: 0.090, green: 0.251, blue: 0.627)
    static let muted = Color(red: 0.078, green: 0.078, blue: 0.078).opacity(0.62)
    static let rule: CGFloat = 2

    static var background: some View { paper.ignoresSafeArea() }

    static func display(_ size: CGFloat, relativeTo style: Font.TextStyle = .title) -> Font {
        .custom("Futura-Bold", size: size, relativeTo: style)
    }

    /// Sequential ramp for attention and activations: paper → yellow → red → black.
    static func heatRGB(_ v: Float) -> (Float, Float, Float) {
        let t = min(max(v, 0), 1)
        let stops: [(Float, Float, Float)] = [(0.949, 0.925, 0.878), (0.957, 0.745, 0.0), (0.835, 0.169, 0.118), (0.078, 0.078, 0.078)]
        let x = t * 3, i = min(Int(x), 2), k = x - Float(i)
        let a = stops[i], b = stops[i + 1]
        return (a.0 + (b.0 - a.0) * k, a.1 + (b.1 - a.1) * k, a.2 + (b.2 - a.2) * k)
    }

    static func heat(_ v: Float) -> Color {
        let c = heatRGB(v)
        return Color(red: Double(c.0), green: Double(c.1), blue: Double(c.2))
    }

    /// Applies Futura to navigation titles so every screen carries the same typographic voice.
    static func configureAppearance() {
        let nav = UINavigationBarAppearance()
        nav.configureWithOpaqueBackground()
        nav.backgroundColor = UIColor(paper)
        nav.shadowColor = UIColor(ink)
        let ink = UIColor(Self.ink)
        if let f = UIFont(name: "Futura-Bold", size: 17) { nav.titleTextAttributes = [.font: f, .foregroundColor: ink] }
        if let f = UIFont(name: "Futura-Bold", size: 32) { nav.largeTitleTextAttributes = [.font: f, .foregroundColor: ink] }
        UINavigationBar.appearance().standardAppearance = nav
        UINavigationBar.appearance().scrollEdgeAppearance = nav
        UINavigationBar.appearance().compactAppearance = nav
    }
}

enum BauhausMark { case circle, square, triangle }

struct MarkShape: View {
    let mark: BauhausMark
    var color: Color
    var size: CGFloat = 12
    var body: some View {
        switch mark {
        case .circle: Circle().fill(color).frame(width: size, height: size)
        case .square: Rectangle().fill(color).frame(width: size, height: size)
        case .triangle: Triangle().fill(color).frame(width: size, height: size)
        }
    }
}

struct Triangle: Shape {
    func path(in r: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: r.minX, y: r.minY)); p.addLine(to: CGPoint(x: r.maxX, y: r.midY)); p.addLine(to: CGPoint(x: r.minX, y: r.maxY))
        p.closeSubpath(); return p
    }
}

/// Flat sheet with a hard black rule and square corners, no shadows or blur.
struct CardModifier: ViewModifier {
    var band: Color?
    func body(content: Content) -> some View {
        content
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.sheet)
            .overlay(alignment: .top) { if let band { Rectangle().fill(band).frame(height: 6) } }
            .overlay(Rectangle().stroke(Theme.ink, lineWidth: Theme.rule))
    }
}

extension View {
    func card(_ band: Color? = nil) -> some View { modifier(CardModifier(band: band)) }
}

struct SectionTitle: View {
    let text: String
    var subtitle: String? = nil
    var mark: BauhausMark = .square
    var color: Color = Theme.red
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                MarkShape(mark: mark, color: color)
                Text(text.uppercased()).font(Theme.display(15, relativeTo: .headline)).tracking(0.6)
            }
            if let subtitle { Text(subtitle).font(.caption).foregroundStyle(Theme.muted) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Primary-colour button used for chips and suggestions.
struct BlockButtonStyle: ButtonStyle {
    var fill: Color = Theme.sheet
    var foreground: Color = Theme.ink
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(foreground)
            .background(configuration.isPressed ? Theme.yellow : fill)
            .overlay(Rectangle().stroke(Theme.ink, lineWidth: Theme.rule))
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
                (r, g, b) = Theme.heatRGB(v)
            case .diverging:
                // blue (negative) ← paper → red (positive)
                let t = min(max(v, -1), 1)
                let p: (Float, Float, Float) = (0.949, 0.925, 0.878)
                let e: (Float, Float, Float) = t < 0 ? (0.090, 0.251, 0.627) : (0.835, 0.169, 0.118)
                let k = abs(t)
                r = p.0 + (e.0 - p.0) * k; g = p.1 + (e.1 - p.1) * k; b = p.2 + (e.2 - p.2) * k
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
            .background(weight > 0.02 ? Theme.heat(weight) : Theme.sheet)
            .overlay(Rectangle().stroke(highlighted ? Theme.red : Theme.ink, lineWidth: highlighted ? 3 : 1.5))
            .foregroundStyle(weight > 0.55 ? Color.white : Theme.ink)
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
                            Rectangle().fill(Theme.ink.opacity(0.06))
                            Rectangle().fill(item.id == shown.first?.id ? Theme.red : Theme.blue)
                                .frame(width: max(4, geo.size.width * CGFloat(item.prob / maxP)))
                        }
                    }
                    .frame(height: 12)
                    Text(item.prob.percent).font(.system(.caption, design: .monospaced)).frame(width: 52, alignment: .trailing)
                        .foregroundStyle(Theme.muted)
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
