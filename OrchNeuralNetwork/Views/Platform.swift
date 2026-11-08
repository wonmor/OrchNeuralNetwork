import SwiftUI

// Small shims so the same views build for iOS and native macOS. iOS behaviour is unchanged.

#if os(iOS)
import UIKit
typealias PlatformImage = UIImage

extension Image {
    init(platformImage: PlatformImage) { self.init(uiImage: platformImage) }
}

extension PlatformImage {
    convenience init(cgImageCompat cg: CGImage) { self.init(cgImage: cg) }
}

enum Haptics {
    static func light() { UIImpactFeedbackGenerator(style: .light).impactOccurred() }
}

extension ToolbarItemPlacement {
    static var trailingBar: ToolbarItemPlacement { .topBarTrailing }
}

extension View {
    func inlineTitle() -> some View { navigationBarTitleDisplayMode(.inline) }
    func plainTextInput() -> some View { autocorrectionDisabled().textInputAutocapitalization(.never) }
    func dismissKeyboardOnScroll() -> some View { scrollDismissesKeyboard(.interactively) }
}

enum PlatformLayout {
    static let isMac = false
    /// Multiplier for canvas heights (heatmaps, neuron grids).
    static let canvas: CGFloat = 1
}

#else
import AppKit
typealias PlatformImage = NSImage

extension Image {
    init(platformImage: PlatformImage) { self.init(nsImage: platformImage) }
}

extension PlatformImage {
    convenience init(cgImageCompat cg: CGImage) { self.init(cgImage: cg, size: NSSize(width: cg.width, height: cg.height)) }
}

enum Haptics {
    static func light() {}
}

extension ToolbarItemPlacement {
    static var trailingBar: ToolbarItemPlacement { .primaryAction }
}

extension View {
    func inlineTitle() -> some View { self }
    func plainTextInput() -> some View { autocorrectionDisabled() }
    func dismissKeyboardOnScroll() -> some View { self }
}

enum PlatformLayout {
    static let isMac = true
    static let canvas: CGFloat = 1.6
}
#endif

extension View {
    /// Mac: sheets need an explicit size; content columns are capped and centred in a wide window.
    @ViewBuilder func macSheetFrame() -> some View {
        if PlatformLayout.isMac { frame(minWidth: 540, idealWidth: 560, minHeight: 640, idealHeight: 720) } else { self }
    }
    @ViewBuilder func readableWidth() -> some View {
        if PlatformLayout.isMac { frame(maxWidth: 980).frame(maxWidth: .infinity) } else { self }
    }
}
