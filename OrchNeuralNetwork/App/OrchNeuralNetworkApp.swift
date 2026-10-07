import SwiftUI

@main
struct OrchNeuralNetworkApp: App {
    @State private var engine = InferenceEngine()

    init() {
        // `-benchmark` runs the technical-report measurements and writes Documents/benchmark.json.
        if CommandLine.arguments.contains("-benchmark") {
            Benchmark.runFromLaunchArgument()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(engine)
                .preferredColorScheme(.dark)
        }
    }
}
