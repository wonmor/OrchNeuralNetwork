import Foundation
import Observation

struct Suggestion: Identifiable, Hashable {
    let id = UUID()
    /// The full word to show in the suggestion bar.
    let display: String
    /// The exact text to append to the input when tapped.
    let insertion: String
    let prob: Float
}

/// Owns the model, debounces input, and publishes the latest trace to the UI.
@Observable
@MainActor
final class InferenceEngine {
    enum State: Equatable { case loading, ready, failed(String) }

    private(set) var state: State = .loading
    private(set) var trace: Trace?
    private(set) var suggestions: [Suggestion] = []
    private(set) var isRunning = false
    var text: String = "" { didSet { if text != oldValue { schedule() } } }

    private(set) var model: GPT2Model?
    private var pending: Task<Void, Never>?
    private var generation = 0

    var parameterCount: Int { model?.file.totalParameters ?? 0 }
    var config: ModelConfig? { model?.config }

    func load() {
        guard model == nil else { return }
        // `xcrun simctl launch <udid> <bundle> -prefill "text"` pre-fills the input (used for screenshots).
        if text.isEmpty, let prefill = UserDefaults.standard.string(forKey: "prefill") { text = prefill }
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let m = try GPT2Model.loadBundled()
                await MainActor.run { [weak self] in
                    self?.model = m
                    self?.state = .ready
                    self?.schedule()
                }
            } catch {
                await MainActor.run { [weak self] in self?.state = .failed(error.localizedDescription) }
            }
        }
    }

    func run(now: Bool = false) { schedule(delay: now ? 0 : 0.25) }

    private func schedule(delay: Double = 0.25) {
        guard let model else { return }
        pending?.cancel()
        generation += 1
        let gen = generation
        let input = text
        pending = Task { [weak self] in
            if delay > 0 { try? await Task.sleep(for: .seconds(delay)) }
            if Task.isCancelled { return }
            await MainActor.run { self?.isRunning = true }
            let result = await Task.detached(priority: .userInitiated) { model.run(text: input) }.value
            guard let self, gen == self.generation else { return }
            self.isRunning = false
            if let result {
                self.trace = result
                self.suggestions = Self.makeSuggestions(from: result, input: input)
            } else {
                self.trace = nil
                self.suggestions = []
            }
        }
    }

    func accept(_ s: Suggestion) {
        text += s.insertion
        run(now: true)
    }

    // MARK: Suggestion logic (what a keyboard would do with the raw distribution)

    nonisolated private static let blocked: Set<String> = ["fuck", "shit", "cunt", "nigger", "faggot", "bitch", "asshole", "dick", "pussy", "rape"]

    nonisolated static func makeSuggestions(from trace: Trace, input: String) -> [Suggestion] {
        let endsWithSpace = input.last?.isWhitespace ?? true
        let lastWord = endsWithSpace ? "" : String(input.split(whereSeparator: { $0.isWhitespace }).last ?? "")
        var seen = Set<String>()
        var out: [Suggestion] = []
        for cand in trace.topK {
            let raw = cand.text
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, trimmed.unicodeScalars.allSatisfy({ CharacterSet.letters.contains($0) || $0 == "'" }) else { continue }
            let startsWord = raw.first == " "
            let display: String
            let insertion: String
            if startsWord {
                display = trimmed
                insertion = endsWithSpace ? trimmed + " " : raw + " "
            } else {
                if endsWithSpace { continue }      // a continuation makes no sense after a space
                display = lastWord + trimmed
                insertion = trimmed + " "
            }
            let key = display.lowercased()
            if seen.contains(key) || blocked.contains(key) { continue }
            seen.insert(key)
            out.append(Suggestion(display: display, insertion: insertion, prob: cand.prob))
            if out.count == 3 { break }
        }
        return out
    }
}
