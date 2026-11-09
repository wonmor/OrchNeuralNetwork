import Foundation
import Observation
import StoreKit

/// The tip jar: three consumable in-app purchases (same ids as the Play products) that unlock nothing.
/// They are tips to the developer, so the app never calls them donations.
enum TipJar {
    static let productIDs = ["orchnn.tip.small", "orchnn.tip.medium", "orchnn.tip.large"]

    static let disclaimer = "Tips are optional, unlock nothing, and are not tax-deductible. Everything in Inside LLM stays free."

    /// Names used when the store has not supplied one (the store's localized name wins).
    static let fallbackNames = [
        "orchnn.tip.small": "Small tip",
        "orchnn.tip.medium": "Medium tip",
        "orchnn.tip.large": "Large tip",
    ]

    static let intro = "Inside LLM is free, with no ads and no tracking. If it has helped you see how language models work, you can leave a tip to support its development."
    static let thanks = "Thank you, truly. Your tip keeps Inside LLM free, offline and open about how it works, and goes into the next things I build."
    static let pending = "Your tip is waiting for approval, for example by a parent through Ask to Buy. Nothing more to do here; if it goes through, thank you."
    static let unavailable = "Store unavailable. Check your connection and try again."
    static let unverified = "The App Store couldn't verify that purchase, so it wasn't completed."

    static func failed(_ reason: String) -> String {
        "The tip didn't go through (\(reason)). Nothing was charged."
    }

    /// Store order: by price, then by the fixed id order.
    static func sorted(_ options: [TipOption]) -> [TipOption] {
        options.sorted {
            $0.price != $1.price ? $0.price < $1.price
                : (productIDs.firstIndex(of: $0.id) ?? 0) < (productIDs.firstIndex(of: $1.id) ?? 0)
        }
    }
}

struct TipOption: Identifiable, Equatable {
    let id: String
    let name: String
    let displayPrice: String
    let price: Decimal
}

enum TipStatus: Equatable {
    case idle
    case purchasing(String)
    case pending
    case thanked
    case failed(String)
}

@MainActor
@Observable
final class TipJarStore {
    enum Load: Equatable { case loading, ready, unavailable }

    private(set) var options: [TipOption] = []
    private(set) var load: Load = .loading
    var status: TipStatus = .idle

    @ObservationIgnored private var products: [String: Product] = [:]
    @ObservationIgnored private var updates: Task<Void, Never>?
    @ObservationIgnored private var started = false

    init() {}

    /// Starts the transaction listener (finishing anything left over from a previous launch) and loads prices.
    func start() {
        guard !started else { return }
        started = true
        #if DEBUG
        if UserDefaults.standard.bool(forKey: "tipPreview") { stagePreview(); return }
        #endif
        updates = Task { [weak self] in
            for await result in Transaction.updates { await self?.handle(update: result) }
        }
        Task {
            for await result in Transaction.unfinished {
                if case .verified(let t) = result { await t.finish() }
            }
            await loadProducts()
        }
    }

    func loadProducts() async {
        load = .loading
        do {
            let found = try await Self.withTimeout(seconds: 25) { try await Product.products(for: TipJar.productIDs) }
            products = Dictionary(uniqueKeysWithValues: found.map { ($0.id, $0) })
            options = TipJar.sorted(found.map {
                TipOption(id: $0.id,
                          name: $0.displayName.isEmpty ? (TipJar.fallbackNames[$0.id] ?? $0.id) : $0.displayName,
                          displayPrice: $0.displayPrice, price: $0.price)
            })
            load = options.isEmpty ? .unavailable : .ready
        } catch {
            load = .unavailable
        }
    }

    struct TimedOut: Error {}

    /// The App Store has no connect step, so the whole price query is bounded instead (25 s, as on Android).
    nonisolated static func withTimeout<T: Sendable>(seconds: Double, _ work: @escaping @Sendable () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await work() }
            group.addTask { try await Task.sleep(for: .seconds(seconds)); throw TimedOut() }
            let first = try await group.next()!
            group.cancelAll()
            return first
        }
    }

    func tip(_ option: TipOption) async {
        guard let product = products[option.id] else {
            #if DEBUG
            if UserDefaults.standard.bool(forKey: "tipPreview") { status = .thanked; return }
            #endif
            status = .failed(TipJar.unavailable)
            return
        }
        status = .purchasing(option.id)
        do {
            switch try await product.purchase() {
            case .success(.verified(let transaction)):
                await transaction.finish()
                status = .thanked
            case .success(.unverified):
                status = .failed(TipJar.unverified)
            case .pending:
                status = .pending
            case .userCancelled:
                status = .idle
            @unknown default:
                status = .idle
            }
        } catch {
            status = .failed(TipJar.failed(Self.reason(error)))
        }
    }

    func dismissMessage() { status = .idle }

    private func handle(update result: VerificationResult<Transaction>) async {
        guard case .verified(let transaction) = result else { return }
        await transaction.finish()
        // A tip that was pending (Ask to Buy) and has now gone through.
        if TipJar.productIDs.contains(transaction.productID), transaction.revocationDate == nil, status == .pending {
            status = .thanked
        }
    }

    private static func reason(_ error: Error) -> String {
        if let e = error as? StoreKitError {
            switch e {
            case .networkError: return "no connection to the App Store"
            case .notAvailableInStorefront: return "not available in your country or region"
            case .notEntitled: return "this Apple Account can't make purchases"
            default: break
            }
        }
        if let e = error as? Product.PurchaseError, e == .purchaseNotAllowed { return "purchases aren't allowed on this device" }
        return "the App Store returned an error"
    }

    #if DEBUG
    /// Screenshot staging (`-tipPreview 1`, optionally `-tipThanks 1`): the US prices set on App Store Connect.
    private func stagePreview() {
        options = zip(TipJar.productIDs, ["$1.99", "$4.99", "$9.99"]).map { id, p in
            TipOption(id: id, name: TipJar.fallbackNames[id]!, displayPrice: p, price: Decimal(string: String(p.dropFirst()))!)
        }
        load = .ready
        if UserDefaults.standard.bool(forKey: "tipThanks") { status = .thanked }
    }
    #endif
}
