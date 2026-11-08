import SwiftUI

/// The tip jar panel shown in About. Prices come from the App Store; nothing here unlocks anything.
struct TipJarPanel: View {
    @Environment(TipJarStore.self) private var store

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: "cup.and.saucer.fill").font(.title3).foregroundStyle(Theme.ink)
                Text("Tip jar").font(Theme.display(20, relativeTo: .title3)).foregroundStyle(Theme.ink)
                    .accessibilityAddTraits(.isHeader)
            }
            Text(TipJar.intro).font(.callout).foregroundStyle(Theme.ink)
                .fixedSize(horizontal: false, vertical: true)

            if store.status == .thanked { thanks } else { choices }

            switch store.status {
            case .pending: message(TipJar.pending, glyph: "hourglass")
            case .failed(let text): message(text, glyph: "exclamationmark.circle")
            default: EmptyView()
            }

            Text(TipJar.disclaimer).font(.caption).foregroundStyle(Theme.muted)
        }
        .card(Theme.blue)
        .task { store.start() }
    }

    @ViewBuilder private var choices: some View {
        switch store.load {
        case .loading:
            HStack(spacing: 10) {
                ProgressView().tint(Theme.ink)
                Text("Loading prices").font(.caption).foregroundStyle(Theme.muted)
            }
            .frame(minHeight: 44)
        case .unavailable:
            VStack(alignment: .leading, spacing: 10) {
                Text(TipJar.unavailable).font(.footnote).foregroundStyle(Theme.muted)
                Button { Task { await store.loadProducts() } } label: {
                    Label("Try again", systemImage: "arrow.clockwise")
                        .font(.subheadline.weight(.semibold)).padding(.horizontal, 14).frame(minHeight: 44)
                }
                .buttonStyle(BlockButtonStyle())
            }
        case .ready:
            VStack(spacing: 10) {
                ForEach(store.options) { option in tipButton(option) }
            }
        }
    }

    private func tipButton(_ option: TipOption) -> some View {
        let busy = store.status == .purchasing(option.id)
        let anyBusy: Bool = { if case .purchasing = store.status { return true } else { return false } }()
        return Button { Task { await store.tip(option) } } label: {
            HStack(spacing: 10) {
                Text(option.name.uppercased()).font(.subheadline.weight(.semibold)).tracking(1.2)
                Spacer(minLength: 8)
                if busy {
                    ProgressView().tint(.white)
                } else {
                    Text(option.displayPrice).font(.subheadline.weight(.semibold)).monospacedDigit()
                }
            }
            .padding(.horizontal, 16)
            .frame(maxWidth: .infinity, minHeight: 52)
        }
        .buttonStyle(BlockButtonStyle(fill: busy ? Theme.blue : Theme.sheet, foreground: busy ? .white : Theme.ink))
        .disabled(anyBusy)
        .accessibilityLabel("\(option.name), \(option.displayPrice)")
        .accessibilityHint("Leaves a tip. Unlocks nothing.")
    }

    private var thanks: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: "heart.fill").font(.largeTitle).foregroundStyle(Theme.red)
                Text("Thank you").font(Theme.display(30, relativeTo: .title)).foregroundStyle(Theme.ink)
                    .accessibilityAddTraits(.isHeader)
            }
            Text(TipJar.thanks).font(.callout).foregroundStyle(Theme.ink)
                .fixedSize(horizontal: false, vertical: true)
            Button { store.dismissMessage() } label: {
                Text("Done").font(.subheadline.weight(.semibold)).padding(.horizontal, 20).frame(minHeight: 44)
            }
            .buttonStyle(BlockButtonStyle())
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.paper)
        .overlay(Rectangle().stroke(Theme.ink, lineWidth: Theme.rule))
    }

    private func message(_ text: String, glyph: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: glyph).foregroundStyle(Theme.red)
            Text(text).font(.footnote).foregroundStyle(Theme.ink)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Button { store.dismissMessage() } label: { Image(systemName: "xmark") }
                .buttonStyle(.plain)
                .frame(width: 44, height: 44)
                .accessibilityLabel("Dismiss")
        }
    }
}

/// The tip jar on its own, opened from the small link at the foot of the Steps tab.
struct TipJarSheet: View {
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background
                ScrollView { TipJarPanel().padding() }
            }
            .navigationTitle("Support")
            .inlineTitle()
            .toolbar { ToolbarItem(placement: .trailingBar) { Button("Done") { dismiss() } } }
        }
    }
}

/// A small, quiet text link. No banner, no badge, no prompt.
struct SupportLink: View {
    @State private var open = false
    var body: some View {
        Button { open = true } label: {
            Label("Support Orch Neural Network", systemImage: "cup.and.saucer")
                .font(.footnote)
                .foregroundStyle(Theme.muted)
                .underline()
                .frame(minHeight: 44)
        }
        .buttonStyle(.plain)
        .accessibilityHint("Opens the tip jar")
        .sheet(isPresented: $open) { TipJarSheet().macSheetFrame() }
    }
}
