import StoreKit
import StoreKitTest
import XCTest
@testable import OrchNeuralNetwork

/// Tip jar against the local StoreKit configuration (StoreKit/TipJar.storekit).
@MainActor
final class TipJarTests: XCTestCase {
    private var session: SKTestSession!

    override func setUp() async throws {
        session = try SKTestSession(configurationFileNamed: "TipJar")
        session.resetToDefaultState()
        session.clearTransactions()
        session.disableDialogs = true
        session.askToBuyEnabled = false
        session.failTransactionsEnabled = false
    }

    override func tearDown() async throws {
        session.clearTransactions()
        session = nil
    }

    func testCopyNeverSaysDonation() {
        for text in [TipJar.disclaimer, TipJar.thanks, TipJar.pending, TipJar.unavailable, TipJar.unverified, TipJar.failed("x")]
            + Array(TipJar.fallbackNames.values) {
            XCTAssertFalse(text.lowercased().contains("donat"), text)
            XCTAssertFalse(text.contains("·") || text.contains("•"), text)
        }
        XCTAssertEqual(TipJar.disclaimer, "Tips are optional, unlock nothing, and are not tax-deductible. Everything in Orch Neural Network stays free.")
        XCTAssertTrue(TipJar.unavailable.hasPrefix("Store unavailable"))
    }

    func testSortingByPrice() {
        let o = { (id: String, p: String) in TipOption(id: id, name: id, displayPrice: p, price: Decimal(string: p)!) }
        let sorted = TipJar.sorted([o("orchnn.tip.large", "9.99"), o("orchnn.tip.small", "1.99"), o("orchnn.tip.medium", "4.99")])
        XCTAssertEqual(sorted.map(\.id), TipJar.productIDs)
    }

    func testProductsLoadAsConsumablesInPriceOrder() async throws {
        let store = TipJarStore()
        await store.loadProducts()
        XCTAssertEqual(store.load, .ready)
        XCTAssertEqual(store.options.map(\.id), TipJar.productIDs)
        XCTAssertEqual(store.options.map(\.name), ["Small tip", "Medium tip", "Large tip"])
        let products = try await Product.products(for: TipJar.productIDs)
        XCTAssertEqual(products.count, 3)
        XCTAssertTrue(products.allSatisfy { $0.type == .consumable })
    }

    func testSuccessfulTipThanksAndFinishes() async throws {
        let store = TipJarStore()
        await store.loadProducts()
        await store.tip(store.options[1])
        XCTAssertEqual(store.status, .thanked)
        var unfinished = 0
        for await _ in Transaction.unfinished { unfinished += 1 }
        XCTAssertEqual(unfinished, 0, "consumable tips must be finished")
        store.dismissMessage()
        XCTAssertEqual(store.status, .idle)
    }

    func testFailedTipShowsMessage() async throws {
        session.failTransactionsEnabled = true
        session.failureError = .unknown
        let store = TipJarStore()
        await store.loadProducts()
        await store.tip(store.options[0])
        guard case .failed(let text) = store.status else { return XCTFail("expected failure, got \(store.status)") }
        XCTAssertTrue(text.contains("Nothing was charged"))
    }

    func testAskToBuyIsPending() async throws {
        session.askToBuyEnabled = true
        let store = TipJarStore()
        await store.loadProducts()
        await store.tip(store.options[0])
        XCTAssertEqual(store.status, .pending)
    }
}
