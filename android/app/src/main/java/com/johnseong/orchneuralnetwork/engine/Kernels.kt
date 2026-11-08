package com.johnseong.orchneuralnetwork.engine

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.Future

/** The two heavy operations of the forward pass, reading float16 weights straight from the mapped file. */
interface Kernels {
    /** c[m, n] = a[m, k] x W[k, n] + bias[n], W row-major float16 at byte [offset] of [weights]. */
    fun matmul(a: FloatArray, m: Int, k: Int, weights: ByteBuffer, offset: Long, n: Int, bias: FloatArray?, c: FloatArray)

    /** out[v] = dot(x, E[v, :]) for the tied embedding matrix E[V, D]. */
    fun logits(x: FloatArray, weights: ByteBuffer, offset: Long, v: Int, d: Int, out: FloatArray)
}

/** NEON / OpenMP kernels in liborchnn (src/main/cpp). Used on devices. */
object NativeKernels : Kernels {
    val available: Boolean = runCatching { System.loadLibrary("orchnn") }.isSuccess

    @JvmStatic private external fun nMatmul(a: FloatArray, m: Int, k: Int, w: ByteBuffer, offset: Long, n: Int, bias: FloatArray?, c: FloatArray)
    @JvmStatic private external fun nLogits(x: FloatArray, w: ByteBuffer, offset: Long, v: Int, d: Int, out: FloatArray)

    override fun matmul(a: FloatArray, m: Int, k: Int, weights: ByteBuffer, offset: Long, n: Int, bias: FloatArray?, c: FloatArray) =
        nMatmul(a, m, k, weights, offset, n, bias, c)

    override fun logits(x: FloatArray, weights: ByteBuffer, offset: Long, v: Int, d: Int, out: FloatArray) =
        nLogits(x, weights, offset, v, d, out)
}

/**
 * Plain Kotlin kernels with the same maths as the native ones. They run the JVM parity tests against the
 * numpy oracle, and are the fallback if the native library cannot load.
 */
class JvmKernels(threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)) : Kernels {
    private val pool = Executors.newFixedThreadPool(threads) { r -> Thread(r, "orchnn-kernel").apply { isDaemon = true } }
    private val parts = threads

    private fun parallel(total: Int, body: (Int, Int) -> Unit) {
        val step = (total + parts - 1) / parts
        val fs = ArrayList<Future<*>>()
        var s = 0
        while (s < total) {
            val a = s; val b = minOf(total, s + step)
            fs += pool.submit { body(a, b) }
            s = b
        }
        fs.forEach { it.get() }
    }

    override fun matmul(a: FloatArray, m: Int, k: Int, weights: ByteBuffer, offset: Long, n: Int, bias: FloatArray?, c: FloatArray) {
        val w = weights.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        parallel(n) { n0, n1 ->
            val width = n1 - n0
            val row = FloatArray(width)
            for (i in 0 until m) for (j in 0 until width) c[i * n + n0 + j] = bias?.get(n0 + j) ?: 0f
            for (kk in 0 until k) {
                val base = offset + 2L * (kk.toLong() * n + n0)
                for (j in 0 until width) row[j] = Half.toFloat(w.getShort((base + 2L * j).toInt()))
                for (i in 0 until m) {
                    val av = a[i * k + kk]
                    val cb = i * n + n0
                    for (j in 0 until width) c[cb + j] += av * row[j]
                }
            }
        }
    }

    override fun logits(x: FloatArray, weights: ByteBuffer, offset: Long, v: Int, d: Int, out: FloatArray) {
        val w = weights.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        parallel(v) { v0, v1 ->
            for (t in v0 until v1) {
                val base = offset + 2L * t * d
                var s = 0f
                for (i in 0 until d) s += x[i] * Half.toFloat(w.getShort((base + 2L * i).toInt()))
                out[t] = s
            }
        }
    }
}
