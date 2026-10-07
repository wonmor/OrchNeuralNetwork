import XCTest
@testable import OrchNeuralNetwork

/// Produces the numbers used in the technical report. Run with
///   xcodebuild test -scheme OrchNeuralNetwork -only-testing:OrchNeuralNetworkTests/BenchmarkTests -destination ...
/// and pick the `ORCH_BENCH_JSON` line out of the log (scripts/run_benchmark.sh does this).
final class BenchmarkTests: XCTestCase {
    func testBenchmark() throws {
        let ref = Bundle(for: Self.self).url(forResource: "reference", withExtension: "json")
        let result = try Benchmark.run(reference: ref)
        let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
        print("ORCH_BENCH_JSON " + String(decoding: data, as: UTF8.self))
        let att = XCTAttachment(data: data, uniformTypeIdentifier: "public.json")
        att.name = "benchmark.json"
        att.lifetime = .keepAlways
        add(att)
        XCTAssertNotNil(result["latency"])
    }
}
