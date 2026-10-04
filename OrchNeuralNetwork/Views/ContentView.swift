import SwiftUI

struct ContentView: View {
    @Environment(InferenceEngine.self) private var engine
    @State private var tab = UserDefaults.standard.integer(forKey: "tab")   // `-tab N` launch argument

    var body: some View {
        TabView(selection: $tab) {
            PredictView()
                .tabItem { Label("Predict", systemImage: "keyboard") }.tag(0)
            InsideView()
                .tabItem { Label("Inside", systemImage: "brain.head.profile") }.tag(1)
            WeightsView()
                .tabItem { Label("Weights", systemImage: "square.grid.3x3.fill") }.tag(2)
            LearnView()
                .tabItem { Label("Learn", systemImage: "book") }.tag(3)
        }
        .tint(Theme.accent)
        .overlay { loadingOverlay }
        .onAppear { engine.load() }
    }

    @ViewBuilder private var loadingOverlay: some View {
        switch engine.state {
        case .ready:
            EmptyView()
        case .loading:
            ZStack {
                Theme.gradient.ignoresSafeArea()
                VStack(spacing: 14) {
                    ProgressView().controlSize(.large)
                    Text("Loading 82 million weights…").font(.headline)
                    Text("Everything runs on this device. Nothing leaves it.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .transition(.opacity)
        case .failed(let message):
            ZStack {
                Theme.gradient.ignoresSafeArea()
                VStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle").font(.largeTitle)
                    Text("Couldn't load the model").font(.headline)
                    Text(message).font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }
                .padding()
            }
        }
    }
}
