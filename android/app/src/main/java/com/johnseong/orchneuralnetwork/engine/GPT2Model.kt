package com.johnseong.orchneuralnetwork.engine

import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.math.tanh

data class TokenProb(val id: Int, val text: String, val prob: Float)

/** Everything recorded for one layer of a forward pass, for visualisation. */
class LayerTrace(
    val id: Int,
    /** attention[head] is a T×T row-major matrix: row = query position, column = key position. */
    val attention: List<FloatArray>,
    /** Top predictions if the model stopped after this layer ("logit lens"). */
    val logitLens: List<TokenProb>,
    /** L2 norm of the last token's residual stream after this layer. */
    val residualNorm: Float,
    /** GELU outputs of the last token's 3072 MLP neurons. */
    val mlpActivations: FloatArray,
    /** Fraction of MLP neurons with activation > 0. */
    val mlpActiveFraction: Float,
)

class Trace(
    val tokenIds: List<Int>,
    val tokens: List<String>,
    val embeddingNorm: Float,
    /** First 64 values of the last token's embedding. */
    val embeddingPreview: FloatArray,
    val layers: List<LayerTrace>,
    val topK: List<TokenProb>,
    val elapsedMilliseconds: Double,
) {
    val tokenCount: Int get() = tokens.size
    val serial: Long = nextSerial()

    private companion object {
        var counter = 0L
        @Synchronized fun nextSerial() = ++counter
    }
}

/**
 * GPT-2 forward pass, a line-for-line port of the iOS engine. No KV cache: every call recomputes the
 * whole sequence so that every intermediate value can be captured. Weights stay float16 in the mapped
 * file; [kernels] widens them on the fly.
 */
class GPT2Model(val file: WeightFile, val tokenizer: BPETokenizer, private val kernels: Kernels) {
    val config: ModelConfig = file.config

    private val raw = file.raw
    private val wteOff = file.offsetOf("wte.weight")
    private val wpe = file.floats("wpe.weight")
    private val lnfG = file.floats("ln_f.weight")
    private val lnfB = file.floats("ln_f.bias")

    private class Block(
        val ln1g: FloatArray, val ln1b: FloatArray,
        val attnW: Long, val attnB: FloatArray,
        val projW: Long, val projB: FloatArray,
        val ln2g: FloatArray, val ln2b: FloatArray,
        val fcW: Long, val fcB: FloatArray,
        val mlpProjW: Long, val mlpProjB: FloatArray,
    )

    private val blocks = (0 until config.nLayer).map { i ->
        val p = "h.$i."
        Block(
            file.floats(p + "ln_1.weight"), file.floats(p + "ln_1.bias"),
            file.offsetOf(p + "attn.c_attn.weight"), file.floats(p + "attn.c_attn.bias"),
            file.offsetOf(p + "attn.c_proj.weight"), file.floats(p + "attn.c_proj.bias"),
            file.floats(p + "ln_2.weight"), file.floats(p + "ln_2.bias"),
            file.offsetOf(p + "mlp.c_fc.weight"), file.floats(p + "mlp.c_fc.bias"),
            file.offsetOf(p + "mlp.c_proj.weight"), file.floats(p + "mlp.c_proj.bias"),
        )
    }

    /**
     * A keyboard predicts the next word after a space, so trailing whitespace is dropped before tokenizing;
     * otherwise GPT-2 turns the lone space into its own token and predicts word fragments.
     */
    fun run(text: String, maxTokens: Int = 48, topK: Int = 12): Trace? {
        var ids = tokenizer.encode(text.trimEnd())
        if (ids.isEmpty()) return null
        if (ids.size > maxTokens) ids = ids.takeLast(maxTokens)
        return run(ids, topK)
    }

    @Synchronized
    fun run(ids: List<Int>, topK: Int = 12): Trace {
        val start = System.nanoTime()
        val T = ids.size; val D = config.nEmbd; val H = config.nHead; val hd = config.headDim
        val F = 4 * D; val V = config.nVocab

        // x[T, D] = wte[id] + wpe[pos]
        val x = FloatArray(T * D)
        for (t in 0 until T) {
            val base = wteOff + 2L * ids[t] * D
            for (i in 0 until D) x[t * D + i] = Half.toFloat(raw.getShort((base + 2L * i).toInt())) + wpe[t * D + i]
        }
        val embeddingNorm = norm(x, (T - 1) * D, D)
        val embeddingPreview = x.copyOfRange((T - 1) * D, (T - 1) * D + 64)

        val layerTraces = ArrayList<LayerTrace>()
        val h = FloatArray(T * D)
        val qkv = FloatArray(T * 3 * D)
        val attnOut = FloatArray(T * D)
        val fc = FloatArray(T * F)
        val tmp = FloatArray(T * D)
        val scale = 1f / sqrt(hd.toFloat())

        for ((li, b) in blocks.withIndex()) {
            // attention
            layerNorm(x, h, T, D, b.ln1g, b.ln1b)
            kernels.matmul(h, T, D, raw, b.attnW, 3 * D, b.attnB, qkv)
            val heads = ArrayList<FloatArray>(H)
            for (head in 0 until H) {
                val s = FloatArray(T * T)
                val qo = head * hd; val ko = D + head * hd; val vo = 2 * D + head * hd
                for (i in 0 until T) for (j in 0..i) {
                    var dot = 0f
                    val qi = i * 3 * D + qo; val kj = j * 3 * D + ko
                    for (e in 0 until hd) dot += qkv[qi + e] * qkv[kj + e]
                    s[i * T + j] = dot * scale
                }
                causalSoftmax(s, T)
                heads += s
                for (i in 0 until T) {
                    val o = i * D + head * hd
                    for (e in 0 until hd) attnOut[o + e] = 0f
                    for (j in 0..i) {
                        val p = s[i * T + j]
                        val vj = j * 3 * D + vo
                        for (e in 0 until hd) attnOut[o + e] += p * qkv[vj + e]
                    }
                }
            }
            kernels.matmul(attnOut, T, D, raw, b.projW, D, b.projB, tmp)
            for (i in 0 until T * D) x[i] += tmp[i]

            // MLP
            layerNorm(x, h, T, D, b.ln2g, b.ln2b)
            kernels.matmul(h, T, D, raw, b.fcW, F, b.fcB, fc)
            gelu(fc)
            val mlpAct = fc.copyOfRange((T - 1) * F, T * F)
            kernels.matmul(fc, T, F, raw, b.mlpProjW, D, b.mlpProjB, tmp)
            for (i in 0 until T * D) x[i] += tmp[i]

            // logit lens on the last token
            val lens = predict(x, T, D, V, 5)
            val active = mlpAct.count { it > 0 }.toFloat() / F
            layerTraces += LayerTrace(li, heads, lens, norm(x, (T - 1) * D, D), mlpAct, active)
        }

        val top = predict(x, T, D, V, topK)
        val ms = (System.nanoTime() - start) / 1e6
        return Trace(ids, ids.map { tokenizer.decode(it) }, embeddingNorm, embeddingPreview, layerTraces, top, ms)
    }

    private val logitsBuf = FloatArray(config.nVocab)

    private fun predict(x: FloatArray, T: Int, D: Int, V: Int, topK: Int): List<TokenProb> {
        val last = x.copyOfRange((T - 1) * D, T * D)
        val normed = FloatArray(D)
        layerNorm(last, normed, 1, D, lnfG, lnfB)
        val l = logitsBuf
        kernels.logits(normed, raw, wteOff, V, D, l)
        var maxv = l[0]
        for (i in 1 until V) if (l[i] > maxv) maxv = l[i]
        var sum = 0.0
        for (i in 0 until V) { val e = exp(l[i] - maxv); l[i] = e; sum += e }
        val inv = (1.0 / sum).toFloat()
        // partial top-k by insertion
        val bi = IntArray(topK + 1); val bp = FloatArray(topK + 1); var n = 0
        for (i in 0 until V) {
            val p = l[i] * inv
            if (n < topK || p > bp[n - 1]) {
                var j = n
                while (j > 0 && bp[j - 1] < p) { bi[j] = bi[j - 1]; bp[j] = bp[j - 1]; j-- }
                bi[j] = i; bp[j] = p
                if (n < topK) n++
            }
        }
        return (0 until n).map { TokenProb(bi[it], tokenizer.decode(bi[it]), bp[it]) }
    }

    private fun layerNorm(x: FloatArray, y: FloatArray, T: Int, D: Int, g: FloatArray, beta: FloatArray) {
        val eps = config.layerNormEpsilon
        for (t in 0 until T) {
            val o = t * D
            var mean = 0f; var meanSq = 0f
            for (i in 0 until D) { val v = x[o + i]; mean += v; meanSq += v * v }
            mean /= D; meanSq /= D
            val variance = maxOf(0f, meanSq - mean * mean) // population variance, as GPT-2 uses
            val inv = 1f / sqrt(variance + eps)
            for (i in 0 until D) y[o + i] = (x[o + i] - mean) * inv * g[i] + beta[i]
        }
    }

    private fun causalSoftmax(s: FloatArray, T: Int) {
        for (i in 0 until T) {
            val row = i * T
            var m = -Float.MAX_VALUE
            for (j in 0..i) m = maxOf(m, s[row + j])
            var sum = 0f
            for (j in 0..i) { val e = exp(s[row + j] - m); s[row + j] = e; sum += e }
            val inv = 1f / sum
            for (j in 0..i) s[row + j] *= inv
            for (j in i + 1 until T) s[row + j] = 0f
        }
    }

    private fun gelu(v: FloatArray) {
        val c = 0.7978845608f // sqrt(2/pi)
        for (i in v.indices) {
            val x = v[i]
            v[i] = 0.5f * x * (1f + tanh(c * (x + 0.044715f * x * x * x)))
        }
    }

    private fun norm(v: FloatArray, offset: Int, count: Int): Float {
        var r = 0f
        for (i in offset until offset + count) r += v[i] * v[i]
        return sqrt(r)
    }
}
