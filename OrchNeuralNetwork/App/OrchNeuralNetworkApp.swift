import SwiftUI

@main
struct OrchNeuralNetworkApp: App {
    @State private var engine = InferenceEngine()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(engine)
                .preferredColorScheme(.dark)
        }
    }
}
