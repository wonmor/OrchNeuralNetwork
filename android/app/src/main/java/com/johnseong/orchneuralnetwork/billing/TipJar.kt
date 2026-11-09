package com.johnseong.orchneuralnetwork.billing

import com.android.billingclient.api.BillingClient.BillingResponseCode
import com.android.billingclient.api.Purchase.PurchaseState

/**
 * The tip jar: three consumable Play Billing products (same ids as the iOS StoreKit products) that unlock
 * nothing. They are tips to the developer, so the app never calls them donations.
 */
object TipJar {
    val productIds = listOf("orchnn.tip.small", "orchnn.tip.medium", "orchnn.tip.large")

    const val DISCLAIMER = "Tips are optional, unlock nothing, and are not tax-deductible. Everything in Inside LLM stays free."
    const val INTRO = "Inside LLM is free, with no ads and no tracking. If it has helped you see how language models work, you can leave a tip to support its development."
    const val THANKS = "Thank you, truly. Your tip keeps Inside LLM free, offline and open about how it works, and goes into the next things I build."
    const val PENDING = "Your tip is pending, for example waiting on a cash payment or a family manager's approval. Nothing more to do here; if it goes through, thank you."
    const val UNAVAILABLE = "Store unavailable. Check your connection and try again."

    val fallbackNames = mapOf(
        "orchnn.tip.small" to "Small tip", "orchnn.tip.medium" to "Medium tip", "orchnn.tip.large" to "Large tip",
    )

    fun failed(reason: String) = "The tip didn't go through ($reason). Nothing was charged."

    fun reason(code: Int): String = when (code) {
        BillingResponseCode.SERVICE_DISCONNECTED, BillingResponseCode.SERVICE_UNAVAILABLE,
        BillingResponseCode.NETWORK_ERROR -> "no connection to Google Play"
        BillingResponseCode.BILLING_UNAVAILABLE -> "Google Play billing isn't available on this device"
        BillingResponseCode.ITEM_UNAVAILABLE -> "this tip isn't available in your country"
        BillingResponseCode.ITEM_ALREADY_OWNED -> "a previous tip is still being completed; try again in a moment"
        else -> "Google Play returned an error"
    }

    /** Store order: by price, then by the fixed id order. */
    fun sorted(options: List<TipOption>) =
        options.sortedWith(compareBy<TipOption>({ it.priceMicros }, { productIds.indexOf(it.id) }))

    /** What the user sees after a purchase flow returns. */
    fun outcome(code: Int, purchaseStates: List<Int>): TipStatus = when (code) {
        BillingResponseCode.OK -> when {
            purchaseStates.any { it == PurchaseState.PURCHASED } -> TipStatus.Thanked
            purchaseStates.any { it == PurchaseState.PENDING } -> TipStatus.Pending
            else -> TipStatus.Idle
        }
        BillingResponseCode.USER_CANCELED -> TipStatus.Idle
        else -> TipStatus.Failed(failed(reason(code)))
    }
}

data class TipOption(val id: String, val name: String, val price: String, val priceMicros: Long)

sealed interface TipStatus {
    data object Idle : TipStatus
    data class Purchasing(val id: String) : TipStatus
    data object Pending : TipStatus
    data object Thanked : TipStatus
    data class Failed(val message: String) : TipStatus
}

enum class TipLoad { Loading, Ready, Unavailable }

/** Everything the tip jar UI draws; the billing client publishes it, screenshots stage it. */
data class TipJarUi(
    val load: TipLoad = TipLoad.Loading,
    val options: List<TipOption> = emptyList(),
    val status: TipStatus = TipStatus.Idle,
    /** Why the tips could not be loaded (Play code / timeout), shown under Unavailable. */
    val detail: String? = null,
) {
    companion object {
        /** Staging for screenshots: the US prices set on Play Console. */
        fun preview(status: TipStatus = TipStatus.Idle) = TipJarUi(TipLoad.Ready,
            TipJar.productIds.zip(listOf(1_990_000L, 4_990_000L, 9_990_000L)).map { (id, m) ->
                TipOption(id, TipJar.fallbackNames.getValue(id), String.format(java.util.Locale.US, "\$%.2f", m / 1e6), m)
            }, status)
    }
}
