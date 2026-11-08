package com.johnseong.orchneuralnetwork

import com.johnseong.orchneuralnetwork.engine.BPETokenizer
import com.johnseong.orchneuralnetwork.engine.GPT2Model
import com.johnseong.orchneuralnetwork.engine.InferenceEngine
import com.johnseong.orchneuralnetwork.engine.JvmKernels
import com.johnseong.orchneuralnetwork.engine.WeightFile
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Compares the Kotlin forward pass against Model/reference.json, written by the independent numpy
 * implementation (Tools/reference.py) from the same weight file. Same tolerances as the iOS test suite.
 */
class ReferenceParityTest {
    companion object {
        private val modelDir: File = listOf(File("../../Model"), File("../Model"), File("Model")).first { File(it, "vocab.json").exists() }
        val model: GPT2Model by lazy {
            GPT2Model(WeightFile.map(File(modelDir, "distilgpt2.tsw")),
                BPETokenizer(File(modelDir, "vocab.json").readText(), File(modelDir, "merges.txt").readText()), JvmKernels())
        }
        val reference: JSONObject by lazy { JSONObject(File(modelDir, "reference.json").readText()) }
    }

    private fun ints(a: org.json.JSONArray) = (0 until a.length()).map { a.getInt(it) }

    @Test fun tokenizerMatchesReference() {
        val prompts = reference.getJSONArray("prompts")
        for (i in 0 until prompts.length()) {
            val p = prompts.getJSONObject(i)
            val ids = ints(p.getJSONArray("token_ids"))
            assertEquals(p.getString("prompt"), ids, model.tokenizer.encode(p.getString("prompt")))
            assertEquals(p.getString("prompt"), model.tokenizer.decode(ids))
        }
    }

    @Test fun forwardPassMatchesReference() {
        val prompts = reference.getJSONArray("prompts")
        for (i in 0 until prompts.length()) {
            val p = prompts.getJSONObject(i)
            val prompt = p.getString("prompt")
            val ids = ints(p.getJSONArray("token_ids"))
            val trace = model.run(ids, topK = 10)
            val top10 = p.getJSONArray("top10")
            assertEquals("top-10 order for $prompt", (0 until top10.length()).map { top10.getJSONObject(it).getInt("id") }, trace.topK.map { it.id })
            for (k in 0 until top10.length()) {
                assertEquals("prob $k for $prompt", top10.getJSONObject(k).getDouble("prob").toFloat(), trace.topK[k].prob, 2e-3f)
            }
            val norms = p.getJSONArray("residual_norms")
            assertEquals(norms.getDouble(0).toFloat(), trace.embeddingNorm, 0.05f)
            val lens = p.getJSONArray("logit_lens")
            for ((li, layer) in trace.layers.withIndex()) {
                val ref = norms.getDouble(li + 1).toFloat()
                assertEquals("norm L$li", ref, layer.residualNorm, ref * 0.01f)
                assertEquals("logit lens L$li for $prompt", lens.getJSONObject(li).getJSONArray("top").getJSONObject(0).getInt("id"), layer.logitLens.first().id)
            }
            val T = ids.size
            val att = p.getJSONObject("attention_last_token")
            for ((key, l, h) in listOf(Triple("layer0_head0", 0, 0), Triple("layer5_head11", 5, 11))) {
                val r = att.getJSONArray(key)
                val row = trace.layers[l].attention[h].copyOfRange((T - 1) * T, T * T)
                for (j in 0 until T) assertEquals("$key[$j]", r.getDouble(j).toFloat(), row[j], 2e-3f)
            }
        }
    }

    @Test fun suggestionsAreWordsOnly() {
        val input = "I am going to the "
        val trace = model.run(input)!!
        val s = InferenceEngine.makeSuggestions(trace, input)
        assertFalse(s.isEmpty())
        assertTrue(s.size <= 3)
        for (x in s) assertTrue(x.display, x.display.all { it.isLetter() || it == '\'' })
    }

    @Test fun midWordSuggestionsCompleteTheWord() {
        val input = "The weather tod"
        val trace = model.run(input)!!
        for (x in InferenceEngine.makeSuggestions(trace, input)) {
            assertTrue(x.insertion, x.insertion.endsWith(" "))
        }
    }
}
