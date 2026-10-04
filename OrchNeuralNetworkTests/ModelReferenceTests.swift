import XCTest
@testable import OrchNeuralNetwork

/// Compares the on-device Swift forward pass against Model/reference.json, which was produced by an
/// independent numpy implementation reading the same weight file.
final class ModelReferenceTests: XCTestCase {
    struct Ref: Decodable {
        struct Prob: Decodable { let id: Int; let token: String; let prob: Float }
        struct Lens: Decodable { let layer: Int; let top: [Prob] }
        struct Prompt: Decodable {
            let prompt: String
            let token_ids: [Int]
            let tokens: [String]
            let top10: [Prob]
            let logit_lens: [Lens]
            let residual_norms: [Float]
            let attention_last_token: [String: [Float]]
        }
        let prompts: [Prompt]
    }

    static let model: GPT2Model = try! GPT2Model.loadBundled()

    func loadReference() throws -> Ref {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "reference", withExtension: "json"))
        return try JSONDecoder().decode(Ref.self, from: Data(contentsOf: url))
    }

    func testTokenizerMatchesReference() throws {
        let ref = try loadReference()
        for p in ref.prompts {
            XCTAssertEqual(Self.model.tokenizer.encode(p.prompt), p.token_ids, p.prompt)
            XCTAssertEqual(Self.model.tokenizer.decode(p.token_ids), p.prompt)
        }
    }

    func testForwardPassMatchesReference() throws {
        let ref = try loadReference()
        for p in ref.prompts {
            let trace = Self.model.run(ids: p.token_ids, topK: 10)
            XCTAssertEqual(trace.topK.map(\.id), p.top10.map(\.id), "top-10 order for \(p.prompt)")
            for (a, b) in zip(trace.topK, p.top10) {
                XCTAssertEqual(a.prob, b.prob, accuracy: 2e-3, "prob of \(b.token) for \(p.prompt)")
            }
            XCTAssertEqual(trace.embeddingNorm, p.residual_norms[0], accuracy: 0.05)
            for (i, layer) in trace.layers.enumerated() {
                XCTAssertEqual(layer.residualNorm, p.residual_norms[i + 1], accuracy: p.residual_norms[i + 1] * 0.01, "norm L\(i)")
                XCTAssertEqual(layer.logitLens.first?.id, p.logit_lens[i].top.first?.id, "logit lens L\(i) for \(p.prompt)")
            }
            let T = p.token_ids.count
            let a00 = Array(trace.layers[0].attention[0][(T - 1) * T ..< T * T])
            let a511 = Array(trace.layers[5].attention[11][(T - 1) * T ..< T * T])
            for (x, y) in zip(a00, p.attention_last_token["layer0_head0"]!) { XCTAssertEqual(x, y, accuracy: 2e-3) }
            for (x, y) in zip(a511, p.attention_last_token["layer5_head11"]!) { XCTAssertEqual(x, y, accuracy: 2e-3) }
        }
    }

    func testSuggestionsAreWordsOnly() {
        let trace = Self.model.run(text: "I am going to the ")!
        let s = InferenceEngine.makeSuggestions(from: trace, input: "I am going to the ")
        XCTAssertFalse(s.isEmpty)
        XCTAssertLessThanOrEqual(s.count, 3)
        for x in s { XCTAssertTrue(x.display.allSatisfy { $0.isLetter || $0 == "'" }, x.display) }
    }

    func testForwardPassPerformance() {
        let ids = Self.model.tokenizer.encode("The quick brown fox jumps over the lazy dog and keeps running through the")
        measure { _ = Self.model.run(ids: ids) }
    }
}
