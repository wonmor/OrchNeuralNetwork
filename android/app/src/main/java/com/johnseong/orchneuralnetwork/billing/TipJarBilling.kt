package com.johnseong.orchneuralnetwork.billing

import android.app.Activity
import android.content.Context
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.android.billingclient.api.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Play Billing for the tip jar. Every purchased tip is consumed (never acknowledged-and-kept), so it can be
 * bought again and grants nothing. Pending purchases are consumed once Play reports them as purchased.
 *
 * Never leaves the sheet on Loading: the connection (15 s) and the product query (25 s) are bounded, a
 * disconnect before setup finishes reports Unavailable, and Try again reconnects from scratch when needed.
 */
class TipJarBilling(context: Context) : PurchasesUpdatedListener {
    var ui by mutableStateOf(TipJarUi())
        private set

    private val appContext = context.applicationContext
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val details = mutableMapOf<String, ProductDetails>()
    private var client = newClient()
    private var connecting = false
    private var generation = 0
    private var loadJob: Job? = null

    private fun newClient() = BillingClient.newBuilder(appContext)
        .setListener(this)
        .enablePendingPurchases(PendingPurchasesParams.newBuilder().enableOneTimeProducts().build())
        .enableAutoServiceReconnection()
        .build()

    fun start() {
        scope.launch {
            if (client.isReady) { if (ui.load != TipLoad.Ready) launchLoad(); return@launch }
            if (connecting) return@launch
            connecting = true
            ui = ui.copy(load = TipLoad.Loading, detail = null)
            val gen = ++generation
            try {
                client.startConnection(object : BillingClientStateListener {
                    override fun onBillingSetupFinished(result: BillingResult) {
                        scope.launch {
                            if (gen != generation) return@launch
                            connecting = false; generation++
                            Log.i(TAG, "setup ${result.responseCode} ${result.debugMessage}")
                            if (result.responseCode != BillingClient.BillingResponseCode.OK) {
                                fail("Google Play billing unavailable (code ${result.responseCode})."); return@launch
                            }
                            withTimeoutOrNull(QUERY_TIMEOUT_MS) { consumeLeftovers() }
                            launchLoad()
                        }
                    }
                    override fun onBillingServiceDisconnected() {
                        scope.launch {
                            Log.w(TAG, "disconnected (connecting=$connecting)")
                            if (connecting && gen == generation) {
                                connecting = false; generation++
                                fail("Lost the connection to Google Play.")
                            }
                        }
                    }
                })
            } catch (e: Exception) {
                connecting = false
                client = newClient()
                fail("Google Play billing unavailable (${e.message})."); return@launch
            }
            delay(CONNECT_TIMEOUT_MS)
            if (connecting && gen == generation) {
                connecting = false; generation++
                runCatching { client.endConnection() }
                client = newClient()
                fail("Google Play did not respond.")
            }
        }
    }

    fun retry() = start()

    private fun fail(why: String) {
        Log.w(TAG, why)
        ui = ui.copy(load = TipLoad.Unavailable, detail = why)
    }

    private fun launchLoad() {
        loadJob?.cancel()
        loadJob = scope.launch {
            ui = ui.copy(load = TipLoad.Loading, detail = null)
            val ok = withTimeoutOrNull(QUERY_TIMEOUT_MS) { loadProducts() }
            if (ok == null) fail("Google Play did not return prices in time.")
        }
    }

    /** PBL 8: one-time products list their purchase options in the List variant; fall back to the old field. */
    private fun offerOf(d: ProductDetails): ProductDetails.OneTimePurchaseOfferDetails? =
        d.oneTimePurchaseOfferDetailsList?.firstOrNull { it.offerId == null }
            ?: d.oneTimePurchaseOfferDetailsList?.firstOrNull()
            ?: d.oneTimePurchaseOfferDetails

    private suspend fun loadProducts(): Boolean {
        val params = QueryProductDetailsParams.newBuilder().setProductList(TipJar.productIds.map {
            QueryProductDetailsParams.Product.newBuilder().setProductId(it).setProductType(BillingClient.ProductType.INAPP).build()
        }).build()
        val r = client.queryProductDetails(params)
        val list = r.productDetailsList.orEmpty()
        details.clear(); list.forEach { details[it.productId] = it }
        val options = list.mapNotNull { d ->
            val offer = offerOf(d) ?: return@mapNotNull null
            TipOption(d.productId, d.name.ifBlank { TipJar.fallbackNames[d.productId] ?: d.productId },
                offer.formattedPrice, offer.priceAmountMicros)
        }
        val code = r.billingResult.responseCode
        val ready = code == BillingClient.BillingResponseCode.OK && options.isNotEmpty()
        ui = ui.copy(load = if (ready) TipLoad.Ready else TipLoad.Unavailable, options = TipJar.sorted(options),
            detail = if (ready) null else "No tips returned by Google Play (code $code).")
        return true
    }

    fun tip(activity: Activity, option: TipOption) {
        val d = details[option.id] ?: run { ui = ui.copy(status = TipStatus.Failed(TipJar.UNAVAILABLE)); return }
        val pdp = BillingFlowParams.ProductDetailsParams.newBuilder().setProductDetails(d)
        offerOf(d)?.offerToken?.let { pdp.setOfferToken(it) }
        val params = BillingFlowParams.newBuilder().setProductDetailsParamsList(listOf(pdp.build())).build()
        ui = ui.copy(status = TipStatus.Purchasing(option.id))
        val result = client.launchBillingFlow(activity, params)
        if (result.responseCode != BillingClient.BillingResponseCode.OK) ui = ui.copy(status = TipJar.outcome(result.responseCode, emptyList()))
    }

    fun dismiss() { ui = ui.copy(status = TipStatus.Idle) }

    override fun onPurchasesUpdated(result: BillingResult, purchases: MutableList<Purchase>?) {
        val list = purchases.orEmpty().filter { p -> p.products.any { it in TipJar.productIds } }
        val status = TipJar.outcome(result.responseCode, list.map { it.purchaseState })
        // A late update for a pending tip (no flow on screen) only upgrades Pending to Thanked.
        if (ui.status is TipStatus.Purchasing || status == TipStatus.Thanked || ui.status == TipStatus.Pending) ui = ui.copy(status = status)
        scope.launch { list.filter { it.purchaseState == Purchase.PurchaseState.PURCHASED }.forEach { consume(it) } }
    }

    /** Tips bought while the app was closed, or still pending at last launch. */
    private suspend fun consumeLeftovers() {
        val r = client.queryPurchasesAsync(QueryPurchasesParams.newBuilder().setProductType(BillingClient.ProductType.INAPP).build())
        r.purchasesList.filter { it.purchaseState == Purchase.PurchaseState.PURCHASED }.forEach { consume(it) }
    }

    private suspend fun consume(p: Purchase) {
        client.consumePurchase(ConsumeParams.newBuilder().setPurchaseToken(p.purchaseToken).build())
    }

    fun close() { scope.cancel(); client.endConnection() }

    private companion object {
        const val TAG = "TipJarBilling"
        const val CONNECT_TIMEOUT_MS = 15_000L
        const val QUERY_TIMEOUT_MS = 25_000L
    }
}
