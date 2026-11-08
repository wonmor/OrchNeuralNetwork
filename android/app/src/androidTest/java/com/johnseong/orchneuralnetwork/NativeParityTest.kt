package com.johnseong.orchneuralnetwork

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.johnseong.orchneuralnetwork.engine.InferenceEngine
import com.johnseong.orchneuralnetwork.engine.NativeKernels
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** The on-device path (memory-mapped asset + native kernels) against the numpy oracle. */
@RunWith(AndroidJUnit4::class)
class NativeParityTest {
    @Test fun nativeForwardPassMatchesReference() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        assertTrue("native kernels must load", NativeKernels.available)
        val model = InferenceEngine.loadBundled(ctx)
        val ref = JSONObject(ctx.assets.open("reference.json").bufferedReader().readText())
        val prompts = ref.getJSONArray("prompts")
        for (i in 0 until prompts.length()) {
            val p = prompts.getJSONObject(i)
            val idsJ = p.getJSONArray("token_ids")
            val ids = (0 until idsJ.length()).map { idsJ.getInt(it) }
            assertEquals(ids, model.tokenizer.encode(p.getString("prompt")))
            val trace = model.run(ids, topK = 10)
            val top = p.getJSONArray("top10")
            for (k in 0 until top.length()) {
                assertEquals(top.getJSONObject(k).getInt("id"), trace.topK[k].id)
                assertEquals(top.getJSONObject(k).getDouble("prob").toFloat(), trace.topK[k].prob, 2e-3f)
            }
            val norms = p.getJSONArray("residual_norms")
            trace.layers.forEachIndexed { li, l ->
                val r = norms.getDouble(li + 1).toFloat()
                assertEquals(r, l.residualNorm, r * 0.01f)
            }
            android.util.Log.i("NativeParity", "${p.getString("prompt")}: ${"%.0f".format(trace.elapsedMilliseconds)} ms")
        }
    }
}
