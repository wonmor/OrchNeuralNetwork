import SwiftUI

@main
struct OrchNeuralNetworkApp: App {
    @State private var engine = InferenceEngine()
    @State private var tips = TipJarStore()
    #if os(macOS)
    @State private var nav = MacNavigation()
    #endif

    init() {
        Theme.configureAppearance()
        // `-benchmark` runs the technical-report measurements and writes Documents/benchmark.json.
        if CommandLine.arguments.contains("-benchmark") {
            Benchmark.runFromLaunchArgument()
        }
    }

    var body: some Scene {
        #if os(macOS)
        WindowGroup {
            MacContentView()
                .environment(engine)
                .environment(tips)
                .environment(nav)
                .task { tips.start() }
                .preferredColorScheme(.light)
                .frame(minWidth: 900, minHeight: 640)
        }
        .defaultSize(width: 1280, height: 860)
        .commands { MacCommands(nav: nav, engine: engine) }
        #else
        WindowGroup {
            ContentView()
                .environment(engine)
                .environment(tips)
                .task { tips.start() }
                .preferredColorScheme(.light)
        }
        #endif
    }
}
