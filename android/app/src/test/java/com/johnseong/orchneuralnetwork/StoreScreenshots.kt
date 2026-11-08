package com.johnseong.orchneuralnetwork

import app.cash.paparazzi.DeviceConfig
import app.cash.paparazzi.Paparazzi
import com.android.resources.Density
import com.android.resources.ScreenOrientation
import com.johnseong.orchneuralnetwork.billing.TipJarUi
import com.johnseong.orchneuralnetwork.billing.TipStatus
import com.johnseong.orchneuralnetwork.engine.InferenceEngine
import com.johnseong.orchneuralnetwork.ui.Staging
import com.johnseong.orchneuralnetwork.ui.TipJarActions
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized

/**
 * Play Store screenshots, rendered with layoutlib from the real composables and a real forward pass of the
 * bundled model (JVM kernels). Run:
 *   ./gradlew :app:recordPaparazziDebug --tests '*StoreScreenshots*'
 * Output: app/src/test/snapshots/images/.
 */
@RunWith(Parameterized::class)
class StoreScreenshots(private val device: String) {
    companion object {
        @JvmStatic @Parameterized.Parameters(name = "{0}")
        fun devices() = listOf("phone", "tablet7", "tablet10")

        private fun config(name: String) = when (name) {
            // 1080 × 2160 keeps within Play's 2:1 limit.
            "phone" -> DeviceConfig.PIXEL_5.copy(screenHeight = 2160, screenWidth = 1080, xdpi = 440, ydpi = 440, density = Density.create(440))
            "tablet7" -> DeviceConfig.NEXUS_7.copy(screenWidth = 1200, screenHeight = 1920, xdpi = 320, ydpi = 320, density = Density.XHIGH,
                orientation = ScreenOrientation.PORTRAIT)
            else -> DeviceConfig.NEXUS_10.copy(screenWidth = 1600, screenHeight = 2560, xdpi = 300, ydpi = 300, density = Density.XHIGH,
                orientation = ScreenOrientation.PORTRAIT)
        }
    }

    @get:Rule val paparazzi = Paparazzi(deviceConfig = config(device), maxPercentDifference = 1.0, useDeviceResolution = true)

    private fun engine(text: String) = InferenceEngine().apply { Staging.still = true; stage(ReferenceParityTest.model, text) }
    private val none = TipJarActions()

    @Test fun s1_predict() = paparazzi.snapshot("1-predict") {
        AppShell(engine("I am going to the "), 0, {}, false, {}, TipJarUi(), none)
    }
    @Test fun s2_inside() = paparazzi.snapshot("2-inside") {
        AppShell(engine("The cat sat on the mat because it was"), 1, {}, false, {}, TipJarUi(), none, insideLayer = 4, insideHead = null)
    }
    @Test fun s3_weights() = paparazzi.snapshot("3-weights") {
        AppShell(engine("I am going to the "), 2, {}, false, {}, TipJarUi(), none, weightsTensor = "h.0.attn.c_attn.weight")
    }
    @Test fun s4_steps() = paparazzi.snapshot("4-steps") {
        AppShell(engine("My favourite food is"), 3, {}, false, {}, TipJarUi(), none)
    }
    @Test fun s5_about() = paparazzi.snapshot("5-about") {
        AppShell(engine("My favourite food is"), 0, {}, true, {}, TipJarUi.preview(), none)
    }
    @Test fun t1_tipjar_thanks() = paparazzi.snapshot("tipjar-thanks") {
        AppShell(engine("My favourite food is"), 0, {}, true, {}, TipJarUi.preview(TipStatus.Thanked), none)
    }
}
