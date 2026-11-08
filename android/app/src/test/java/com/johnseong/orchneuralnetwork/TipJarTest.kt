package com.johnseong.orchneuralnetwork

import com.android.billingclient.api.BillingClient.BillingResponseCode
import com.android.billingclient.api.Purchase.PurchaseState
import com.johnseong.orchneuralnetwork.billing.TipJar
import com.johnseong.orchneuralnetwork.billing.TipJarUi
import com.johnseong.orchneuralnetwork.billing.TipOption
import com.johnseong.orchneuralnetwork.billing.TipStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TipJarTest {
    @Test fun outcomes() {
        assertEquals(TipStatus.Thanked, TipJar.outcome(BillingResponseCode.OK, listOf(PurchaseState.PURCHASED)))
        assertEquals(TipStatus.Pending, TipJar.outcome(BillingResponseCode.OK, listOf(PurchaseState.PENDING)))
        assertEquals(TipStatus.Idle, TipJar.outcome(BillingResponseCode.USER_CANCELED, emptyList()))
        val failed = TipJar.outcome(BillingResponseCode.NETWORK_ERROR, emptyList())
        assertTrue(failed is TipStatus.Failed && failed.message.contains("Nothing was charged"))
        assertTrue(TipJar.outcome(BillingResponseCode.ERROR, emptyList()) is TipStatus.Failed)
    }

    @Test fun sortedByPrice() {
        val o = { id: String, m: Long -> TipOption(id, id, "", m) }
        assertEquals(TipJar.productIds, TipJar.sorted(listOf(o("orchnn.tip.large", 9_990_000), o("orchnn.tip.small", 1_990_000),
            o("orchnn.tip.medium", 4_990_000))).map { it.id })
    }

    @Test fun copyNeverSaysDonationOrUsesDots() {
        val all = listOf(TipJar.DISCLAIMER, TipJar.INTRO, TipJar.THANKS, TipJar.PENDING, TipJar.UNAVAILABLE, TipJar.failed("x")) +
            TipJar.fallbackNames.values
        all.forEach { assertFalse(it, it.lowercase().contains("donat") || it.contains("·") || it.contains("•")) }
        assertTrue(TipJar.UNAVAILABLE.startsWith("Store unavailable"))
        assertEquals(listOf("$1.99", "$4.99", "$9.99"), TipJarUi.preview().options.map { it.price })
    }
}
