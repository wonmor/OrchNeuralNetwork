import Foundation
import Darwin

/// Reproducible measurements for the technical report: load time, forward-pass latency as a function of
/// sequence length, memory footprint, and (when a reference file is supplied) agreement with the numpy oracle.
///
/// Two entry points share this code: `BenchmarkTests` in the test bundle (simulator or device via Xcode),
/// and the launch argument `-benchmark` on the app itself, which writes `benchmark.json` into the app's
/// Documents folder so it can be pulled off a physical phone with Finder or `devicectl`.
enum Benchmark {
    static let sequenceLengths = [1, 2, 4, 8, 16, 32, 48]
    static let warmup = 3
    static let repeats = 20

    /// Long enough to yield at least 48 tokens.
    static let corpus = """
    The quick brown fox jumps over the lazy dog while the farmer, who had been awake since before dawn, \
    watched from the porch with a cup of coffee and wondered whether the rain would arrive before the \
    harvest was finished, as the forecast on the radio had promised twice already that week without \
    being right either time.
    """

    static func run(reference: URL?) throws -> [String: Any] {
        var out: [String: Any] = [:]
        out["device"] = deviceInfo()
        out["timestamp"] = ISO8601DateFormatter().string(from: Date())

        // --- load ---
        let footprintBefore = physFootprint()
        let t0 = now()
        let bundle = Bundle.main
        guard let w = bundle.url(forResource: "distilgpt2", withExtension: "tsw"),
              let v = bundle.url(forResource: "vocab", withExtension: "json"),
              let m = bundle.url(forResource: "merges", withExtension: "txt") else {
            throw WeightFileError.missingTensor("bundle resources")
        }
        let file = try WeightFile(url: w)
        let tMapped = now()
        let tok = try BPETokenizer(vocabURL: v, mergesURL: m)
        let tTok = now()
        let model = try GPT2Model(file: file, tokenizer: tok)
        let tModel = now()
        let footprintAfterLoad = physFootprint()
        let fileSize = (try? FileManager.default.attributesOfItem(atPath: w.path)[.size] as? Int) ?? 0
        out["load"] = [
            "weight_file_bytes": fileSize,
            "tensors": file.tensors.count,
            "parameters": file.totalParameters,
            "map_and_parse_header_ms": ms(t0, tMapped),
            "tokenizer_ms": ms(tMapped, tTok),
            "fp16_to_fp32_expand_ms": ms(tTok, tModel),
            "total_ms": ms(t0, tModel),
            "footprint_before_bytes": footprintBefore,
            "footprint_after_load_bytes": footprintAfterLoad,
        ]

        // --- tokenizer throughput ---
        let allIDs = tok.encode(corpus)
        precondition(allIDs.count >= sequenceLengths.max()!, "corpus too short: \(allIDs.count) tokens")
        var tokTimes: [Double] = []
        for _ in 0 ..< repeats { let a = now(); _ = tok.encode(corpus); tokTimes.append(ms(a, now())) }
        out["tokenizer"] = ["corpus_tokens": allIDs.count, "corpus_chars": corpus.count, "encode_ms": stats(tokTimes)]

        // --- forward latency vs T ---
        var latency: [[String: Any]] = []
        var peakFootprint = footprintAfterLoad
        for T in sequenceLengths {
            let ids = Array(allIDs.prefix(T))
            for _ in 0 ..< warmup { _ = model.run(ids: ids) }
            var times: [Double] = []
            var wall: [Double] = []
            for _ in 0 ..< repeats {
                let a = now()
                let tr = model.run(ids: ids)
                wall.append(ms(a, now()))
                times.append(tr.elapsedMilliseconds)
            }
            peakFootprint = max(peakFootprint, physFootprint())
            latency.append(["T": T, "forward_ms": stats(times), "wall_ms": stats(wall)])
        }
        out["latency"] = latency
        out["footprint_peak_bytes"] = peakFootprint
        out["thermal_state_end"] = thermalState()

        // --- agreement with the numpy oracle ---
        if let reference, let data = try? Data(contentsOf: reference),
           let ref = try? JSONDecoder().decode(Reference.self, from: data) {
            out["agreement"] = agreement(model: model, ref: ref)
        }
        return out
    }

    // MARK: Oracle comparison

    struct Reference: Decodable {
        struct Prob: Decodable { let id: Int; let token: String; let prob: Float }
        struct Lens: Decodable { let layer: Int; let top: [Prob] }
        struct Prompt: Decodable {
            let prompt: String
            let token_ids: [Int]
            let top10: [Prob]
            let logit_lens: [Lens]
            let residual_norms: [Float]
            let attention_last_token: [String: [Float]]
        }
        let prompts: [Prompt]
    }

    static func agreement(model: GPT2Model, ref: Reference) -> [String: Any] {
        var rows: [[String: Any]] = []
        var maxProbErr: Float = 0, maxNormRel: Float = 0, maxAttnErr: Float = 0
        var top10Exact = 0, lensAgree = 0, lensTotal = 0, tokOK = 0
        for p in ref.prompts {
            let ids = model.tokenizer.encode(p.prompt)
            let tokMatch = ids == p.token_ids
            if tokMatch { tokOK += 1 }
            let tr = model.run(ids: p.token_ids, topK: 10)
            let orderMatch = tr.topK.map(\.id) == p.top10.map(\.id)
            if orderMatch { top10Exact += 1 }
            var probErr: Float = 0
            for (a, b) in zip(tr.topK, p.top10) { probErr = max(probErr, abs(a.prob - b.prob)) }
            var normRel: Float = abs(tr.embeddingNorm - p.residual_norms[0]) / p.residual_norms[0]
            var lensHits = 0
            for (i, l) in tr.layers.enumerated() {
                normRel = max(normRel, abs(l.residualNorm - p.residual_norms[i + 1]) / p.residual_norms[i + 1])
                if l.logitLens.first?.id == p.logit_lens[i].top.first?.id { lensHits += 1 }
            }
            lensAgree += lensHits; lensTotal += tr.layers.count
            let T = p.token_ids.count
            var attnErr: Float = 0
            let a00 = Array(tr.layers[0].attention[0][(T - 1) * T ..< T * T])
            let a511 = Array(tr.layers[5].attention[11][(T - 1) * T ..< T * T])
            for (x, y) in zip(a00, p.attention_last_token["layer0_head0"] ?? []) { attnErr = max(attnErr, abs(x - y)) }
            for (x, y) in zip(a511, p.attention_last_token["layer5_head11"] ?? []) { attnErr = max(attnErr, abs(x - y)) }
            maxProbErr = max(maxProbErr, probErr); maxNormRel = max(maxNormRel, normRel); maxAttnErr = max(maxAttnErr, attnErr)
            rows.append(["prompt": p.prompt, "tokens": T, "tokenizer_match": tokMatch, "top10_order_match": orderMatch,
                         "max_abs_prob_err": probErr, "max_rel_residual_norm_err": normRel,
                         "logit_lens_top1_agree": lensHits, "max_abs_attention_err": attnErr,
                         "top1_swift": tr.topK.first?.text ?? "", "top1_prob_swift": tr.topK.first?.prob ?? 0,
                         "top1_prob_ref": p.top10.first?.prob ?? 0])
        }
        return ["prompts": rows, "n_prompts": ref.prompts.count, "tokenizer_match": tokOK, "top10_order_exact": top10Exact,
                "logit_lens_top1_agree": lensAgree, "logit_lens_total": lensTotal,
                "max_abs_prob_err": maxProbErr, "max_rel_residual_norm_err": maxNormRel, "max_abs_attention_err": maxAttnErr]
    }

    // MARK: Helpers

    static func now() -> Double { CFAbsoluteTimeGetCurrent() }
    static func ms(_ a: Double, _ b: Double) -> Double { (b - a) * 1000 }

    static func stats(_ xs: [Double]) -> [String: Double] {
        let s = xs.sorted()
        func q(_ f: Double) -> Double { s[min(s.count - 1, Int((Double(s.count - 1) * f).rounded()))] }
        let mean = s.reduce(0, +) / Double(s.count)
        let sd = (s.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(max(1, s.count - 1))).squareRoot()
        return ["n": Double(s.count), "min": s.first!, "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": s.last!, "mean": mean, "sd": sd]
    }

    static func physFootprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint) : -1
    }

    static func thermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"; case .fair: return "fair"; case .serious: return "serious"; case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static func deviceInfo() -> [String: Any] {
        var sys = utsname(); uname(&sys)
        let machine = withUnsafePointer(to: &sys.machine) { $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) } }
        let pi = ProcessInfo.processInfo
        #if targetEnvironment(simulator)
        let sim = true
        let host = pi.environment["SIMULATOR_MODEL_IDENTIFIER"] ?? ""
        #else
        let sim = false
        let host = ""
        #endif
        var brand = [CChar](repeating: 0, count: 256); var size = brand.count
        sysctlbyname("machdep.cpu.brand_string", &brand, &size, nil, 0)
        return ["machine": machine, "simulator": sim, "simulated_model": host, "host_cpu": String(cString: brand),
                "os_version": pi.operatingSystemVersionString, "processor_count": pi.processorCount,
                "active_processor_count": pi.activeProcessorCount, "physical_memory_bytes": Int(pi.physicalMemory),
                "thermal_state_start": thermalState(), "build": optimisationNote()]
    }

    static func optimisationNote() -> String {
        #if DEBUG
        return "Debug configuration (-O -Ounchecked per project.yml)"
        #else
        return "Release configuration"
        #endif
    }

    /// Runs the benchmark, prints the JSON on one marked line, and writes it to Documents. Used by `-benchmark`.
    static func runFromLaunchArgument() {
        let ref = Bundle.main.url(forResource: "reference", withExtension: "json")
        do {
            let result = try run(reference: ref)
            let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
            print("ORCH_BENCH_JSON " + String(decoding: data, as: UTF8.self))
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let pretty = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys, .prettyPrinted])
            try pretty.write(to: docs.appendingPathComponent("benchmark.json"))
            print("ORCH_BENCH_DONE \(docs.appendingPathComponent("benchmark.json").path)")
        } catch {
            print("ORCH_BENCH_ERROR \(error)")
        }
    }
}
