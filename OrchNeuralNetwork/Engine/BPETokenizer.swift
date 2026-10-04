import Foundation

/// GPT-2 style byte-level Byte Pair Encoding tokenizer.
final class BPETokenizer {
    private let encoder: [String: Int]
    private let decoder: [Int: String]
    private let bpeRanks: [String: Int]
    private let byteEncoder: [UInt8: Character]
    private let byteDecoder: [Character: UInt8]
    private let pattern: NSRegularExpression
    private var cache: [String: [String]] = [:]
    private let lock = NSLock()

    var vocabularySize: Int { encoder.count }

    init(vocabURL: URL, mergesURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
        encoder = vocab
        decoder = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        var ranks: [String: Int] = [:]
        var rank = 0
        for line in mergesText.split(separator: "\n", omittingEmptySubsequences: true) {
            if line.hasPrefix("#") { continue }
            let parts = line.split(separator: " ")
            guard parts.count == 2 else { continue }
            ranks["\(parts[0]) \(parts[1])"] = rank
            rank += 1
        }
        bpeRanks = ranks

        var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var be: [UInt8: Character] = [:]
        var bd: [Character: UInt8] = [:]
        for (b, c) in zip(bs, cs) {
            let ch = Character(UnicodeScalar(c)!)
            be[UInt8(b)] = ch
            bd[ch] = UInt8(b)
        }
        byteEncoder = be
        byteDecoder = bd

        pattern = try NSRegularExpression(
            pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)
    }

    // MARK: Encoding

    func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        let ns = text as NSString
        for match in pattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let piece = ns.substring(with: match.range)
            let mapped = String(piece.utf8.map { byteEncoder[$0]! })
            for token in bpe(mapped) {
                if let id = encoder[token] { ids.append(id) }
            }
        }
        return ids
    }

    private func bpe(_ token: String) -> [String] {
        lock.lock()
        if let cached = cache[token] { lock.unlock(); return cached }
        lock.unlock()

        var word = token.map { String($0) }
        guard word.count > 1 else { return word }
        while true {
            var best: (pair: (String, String), rank: Int)? = nil
            for i in 0 ..< word.count - 1 {
                if let r = bpeRanks["\(word[i]) \(word[i + 1])"], best == nil || r < best!.rank {
                    best = ((word[i], word[i + 1]), r)
                }
            }
            guard let (first, second) = best?.pair else { break }
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1, word[i] == first, word[i + 1] == second {
                    merged.append(first + second)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
            if word.count == 1 { break }
        }
        lock.lock(); cache[token] = word; lock.unlock()
        return word
    }

    // MARK: Decoding

    /// Human-readable text for a token id (byte-level symbols mapped back; lossy for partial UTF-8).
    func decode(_ id: Int) -> String {
        guard let token = decoder[id] else { return "�" }
        let bytes = token.compactMap { byteDecoder[$0] }
        return String(decoding: bytes, as: UTF8.self)
    }

    func decode(_ ids: [Int]) -> String {
        let bytes = ids.flatMap { id -> [UInt8] in
            guard let token = decoder[id] else { return [] }
            return token.compactMap { byteDecoder[$0] }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
