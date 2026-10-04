import SwiftUI

/// A guided walkthrough of one forward pass, using the user's own live input.
struct LearnView: View {
    @Environment(InferenceEngine.self) private var engine

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.gradient.ignoresSafeArea()
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        intro
                        if let trace = engine.trace {
                            step1(trace); step2(trace); step3(trace); step4(trace); step5(trace); step6(trace)
                        } else {
                            Text("Type something in the Predict tab and come back: every step below will use your words.")
                                .font(.footnote).foregroundStyle(.secondary).card()
                        }
                        outro
                    }
                    .padding()
                }
            }
            .navigationTitle("How it works")
        }
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("A language model is a next-word predictor").font(.title3.weight(.semibold))
            Text("Your phone keyboard, ChatGPT and this app all do the same core thing: turn text into numbers, push the numbers through a stack of layers, and read off a probability for every possible next token. Then they do it again for the next word. Here is one pass, step by step, on your own text.")
                .font(.footnote).foregroundStyle(.secondary)
        }
        .card()
    }

    private func step(_ n: Int, _ title: String, _ body: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("\(n)").font(.caption.weight(.bold)).frame(width: 22, height: 22)
                    .background(Theme.accent, in: Circle()).foregroundStyle(.black)
                Text(title).font(.headline)
            }
            content()
            Text(body).font(.footnote).foregroundStyle(.secondary)
        }
        .card()
    }

    private func step1(_ t: Trace) -> some View {
        step(1, "Tokenize",
             "Text is chopped into tokens, pieces of words from a fixed list of 50,257. Common words are one token, rare ones are several. Your input became \(t.tokenCount) tokens. ␣ marks a leading space, which is part of the token.") {
            TokenStrip(tokens: t.tokens)
        }
    }

    private func step2(_ t: Trace) -> some View {
        step(2, "Embed",
             "Each token id looks up a row in a 50,257 × 768 table, giving a vector of 768 numbers. A second table adds the token's position. Shown: the first 64 numbers of the last token's vector. The model never sees letters again, only these vectors.") {
            HStack(alignment: .center, spacing: 1) {
                ForEach(Array(t.embeddingPreview.enumerated()), id: \.offset) { _, v in
                    Rectangle()
                        .fill(v >= 0 ? Theme.amber : Theme.accent)
                        .frame(height: CGFloat(min(abs(v) * 60, 40)) + 1)
                        .frame(maxWidth: .infinity)
                }
            }
            .frame(height: 44, alignment: .center)
        }
    }

    private func step3(_ t: Trace) -> some View {
        let l0 = t.layers[0]
        let T = t.tokenCount
        var avg = [Float](repeating: 0, count: T * T)
        for h in l0.attention { for i in 0 ..< T * T { avg[i] += h[i] / Float(l0.attention.count) } }
        return step(3, "Attention: tokens look at each other",
                    "In every layer, each token computes a query, and every earlier token offers a key. Where they match, information flows. Twelve heads do this in parallel, each learning its own habit: one tracks the previous word, another finds the subject of the sentence. This is layer 0, averaged over heads.") {
            AttentionHeatmap(matrix: avg, tokens: t.tokens).frame(height: min(260, CGFloat(T) * 20 + 40))
        }
    }

    private func step4(_ t: Trace) -> some View {
        let last = t.layers[t.layers.count - 1]
        return step(4, "MLP: each token thinks on its own",
                    "After attention, each token's vector is expanded to 3,072 numbers, passed through a nonlinearity, and squeezed back to 768. This is where much of the model's stored knowledge lives. In the last layer \(Int(last.mlpActiveFraction * 100))% of neurons fired for your last token.") {
            NeuronGrid(activations: last.mlpActivations).frame(height: 100)
        }
    }

    private func step5(_ t: Trace) -> some View {
        step(5, "Six layers, one refined guess",
             "Attention and MLP repeat six times. The 'logit lens' reads a prediction after every layer, so you can see the answer form. Early layers often make a crude guess; later layers refine or overturn it.") {
            VStack(spacing: 4) {
                ForEach(t.layers) { l in
                    HStack {
                        Text("Layer \(l.id)").font(.caption.monospaced()).frame(width: 60, alignment: .leading)
                        ForEach(l.logitLens.prefix(3)) { p in
                            TokenChip(text: p.text, weight: p.prob)
                        }
                        Spacer()
                        Text(l.logitLens.first?.prob.percent ?? "").font(.caption.monospaced()).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private func step6(_ t: Trace) -> some View {
        step(6, "Read off probabilities",
             "The final vector is compared with all 50,257 token embeddings to score each one, and a softmax turns the scores into probabilities. A keyboard just shows the top three words. A chatbot samples one, appends it, and runs the whole thing again.") {
            ProbabilityBars(items: t.topK, limit: 5)
        }
    }

    private var outro: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Why can't I see every weight at once?").font(.headline)
            Text("This small model has \(engine.parameterCount.grouped) weights. A 4K screen has about 8 million pixels. Frontier models have hundreds of billions of weights, so any picture of a network is a sample. The Weights tab lets you page through the real matrices, one at a time.")
                .font(.footnote).foregroundStyle(.secondary)
        }
        .card()
    }
}
