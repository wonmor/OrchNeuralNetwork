import SwiftUI

struct AboutView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(InferenceEngine.self) private var engine

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Inside LLM").font(Theme.display(24, relativeTo: .title2))
                            Text("See inside a language model while it predicts your next word.")
                                .font(.footnote).foregroundStyle(Theme.muted)
                        }
                        .card()
                        TipJarPanel()
                        VStack(alignment: .leading, spacing: 6) {
                            Text("The model").font(.headline)
                            Text("DistilGPT2 by Hugging Face: a 6-layer, 82-million-parameter distillation of OpenAI's GPT-2. Weights are stored on the device as 16-bit floats and computed in 32-bit using Apple's Accelerate framework. The whole network is re-run from scratch on every keystroke so that every intermediate value can be shown.")
                                .font(.footnote).foregroundStyle(Theme.muted)
                            if let c = engine.config {
                                Text("\(c.n_layer) layers  |  \(c.n_head) heads  |  \(c.n_embd)-dim  |  \(c.n_vocab.grouped) tokens  |  \(engine.parameterCount.grouped) parameters")
                                    .font(.caption.monospaced()).foregroundStyle(Theme.muted)
                            }
                            Link("Model card and Apache 2.0 licence", destination: URL(string: "https://huggingface.co/distilgpt2")!)
                                .font(.footnote)
                        }
                        .card()
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Privacy").font(.headline)
                            Text("Everything you type is processed on your device and never leaves it. No accounts, no analytics, no ads. The only network use is the App Store loading tip prices for the tip jar.")
                                .font(.footnote).foregroundStyle(Theme.muted)
                        }
                        .card()
                        VStack(alignment: .leading, spacing: 6) {
                            Text("A note on predictions").font(.headline)
                            Text("GPT-2 was trained on web text from 2019. Its suggestions reflect that data, can be wrong, dated or odd, and are shown for learning purposes only.")
                                .font(.footnote).foregroundStyle(Theme.muted)
                        }
                        .card()
                    }
                    .padding()
                }
            }
            .navigationTitle("About")
            .inlineTitle()
            .toolbar { ToolbarItem(placement: .trailingBar) { Button("Done") { dismiss() } } }
        }
    }
}
