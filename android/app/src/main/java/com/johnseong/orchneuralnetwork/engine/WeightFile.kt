package com.johnseong.orchneuralnetwork.engine

import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/** One tensor inside a `.tsw` weight file. */
data class TensorInfo(val name: String, val shape: List<Int>, val dtype: String, val offset: Long, val nbytes: Long) {
    val count: Int get() = shape.fold(1) { a, b -> a * b }
    val shapeDescription: String get() = shape.joinToString(" × ")
}

data class ModelConfig(
    val nLayer: Int, val nHead: Int, val nEmbd: Int, val nCtx: Int, val nVocab: Int, val layerNormEpsilon: Float,
) {
    val headDim: Int get() = nEmbd / nHead
}

class WeightFileException(message: String) : Exception(message)

/**
 * Memory-mapped reader for the app's compact fp16 weight container, the same file the iOS app ships.
 *
 * Layout: "TSW1" magic, UInt32 LE header length, UTF-8 JSON header, zero padding to a 64-byte boundary,
 * then raw little-endian float16 tensors at 64-byte-aligned offsets.
 */
class WeightFile(private val buffer: ByteBuffer) {
    val model: String
    val config: ModelConfig
    val tensors: List<TensorInfo>
    /** Byte offset of the first tensor inside [buffer]. */
    val dataStart: Long
    private val index: Map<String, TensorInfo>

    val totalParameters: Long get() = tensors.sumOf { it.count.toLong() }

    init {
        val b = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        if (b.capacity() < 8 || b.get(0) != 'T'.code.toByte() || b.get(1) != 'S'.code.toByte() ||
            b.get(2) != 'W'.code.toByte() || b.get(3) != '1'.code.toByte()) throw WeightFileException("The weight file has an unexpected format.")
        val headerLength = b.getInt(4).toLong() and 0xffffffffL
        if (b.capacity() < 8 + headerLength) throw WeightFileException("The weight file is incomplete.")
        val bytes = ByteArray(headerLength.toInt())
        b.position(8); b.get(bytes)
        val json = JSONObject(String(bytes, Charsets.UTF_8))
        model = json.getString("model")
        val c = json.getJSONObject("config")
        config = ModelConfig(c.getInt("n_layer"), c.getInt("n_head"), c.getInt("n_embd"), c.getInt("n_ctx"),
            c.getInt("n_vocab"), c.getDouble("layer_norm_epsilon").toFloat())
        val arr = json.getJSONArray("tensors")
        tensors = (0 until arr.length()).map { i ->
            val t = arr.getJSONObject(i)
            val s = t.getJSONArray("shape")
            TensorInfo(t.getString("name"), (0 until s.length()).map { s.getInt(it) }, t.getString("dtype"),
                t.getLong("offset"), t.getLong("nbytes"))
        }
        dataStart = (8 + headerLength + 63) / 64 * 64
        index = tensors.associateBy { it.name }
        for (t in tensors) if (dataStart + t.offset + t.nbytes > b.capacity()) throw WeightFileException("The weight file is incomplete.")
    }

    /** The mapped file itself (direct buffer) for the native kernels. */
    val raw: ByteBuffer get() = buffer

    fun info(name: String): TensorInfo = index[name] ?: throw WeightFileException("Tensor $name is missing from the weight file.")

    /** Absolute byte offset of a float16 tensor inside [raw]. */
    fun offsetOf(name: String): Long {
        val t = info(name)
        if (t.dtype != "f16") throw WeightFileException("Unsupported tensor type ${t.dtype}.")
        return dataStart + t.offset
    }

    fun half(i: Long): Short = buffer.getShort(i.toInt())

    /** Loads a tensor as float32. Used for the small vectors (biases, norms, positions). */
    fun floats(name: String): FloatArray {
        val off = offsetOf(name)
        val n = info(name).count
        val sb = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(n) { Half.toFloat(sb.getShort((off + 2L * it).toInt())) }
    }

    /** Samples a 2-D tensor on a coarse grid without materialising it. Row-major `rows × cols`. */
    fun sampleGrid(name: String, rows: Int, cols: Int): Triple<FloatArray, Int, Int> {
        val t = info(name)
        val h = if (t.shape.size >= 2) t.shape[0] else 1
        val w = if (t.shape.size >= 2) t.shape.drop(1).fold(1) { a, b -> a * b } else t.shape[0]
        val rs = rows.coerceIn(1, h); val cs = cols.coerceIn(1, w)
        val off = offsetOf(name)
        val sb = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(rs * cs)
        for (r in 0 until rs) {
            val srcRow = r.toLong() * h / rs
            for (c in 0 until cs) {
                val srcCol = c.toLong() * w / cs
                out[r * cs + c] = Half.toFloat(sb.getShort((off + 2 * (srcRow * w + srcCol)).toInt()))
            }
        }
        return Triple(out, rs, cs)
    }

    companion object {
        fun map(file: File): WeightFile = FileInputStream(file).use { s ->
            WeightFile(s.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length()).order(ByteOrder.LITTLE_ENDIAN))
        }

        /** Maps a slice of a file, e.g. an uncompressed asset at [start] inside the APK. */
        fun map(fd: java.io.FileDescriptor, start: Long, length: Long): WeightFile = FileInputStream(fd).use { s ->
            WeightFile(s.channel.map(FileChannel.MapMode.READ_ONLY, start, length).order(ByteOrder.LITTLE_ENDIAN))
        }
    }
}

/** IEEE 754 binary16 to float32 via a lookup table (Float.float16ToFloat needs Java 20 / API 35). */
object Half {
    private val table = FloatArray(65536) { slow(it) }
    fun toFloat(h: Short): Float = table[h.toInt() and 0xffff]
    private fun slow(h: Int): Float {
        val sign = (h and 0x8000) shl 16
        var exp = (h ushr 10) and 0x1f
        var man = h and 0x3ff
        val bits = when {
            exp == 0 && man == 0 -> sign
            exp == 0 -> {
                exp = 127 - 15 + 1
                while (man and 0x400 == 0) { man = man shl 1; exp-- }
                sign or (exp shl 23) or ((man and 0x3ff) shl 13)
            }
            exp == 31 -> sign or 0x7f800000 or (man shl 13)
            else -> sign or ((exp + 127 - 15) shl 23) or (man shl 13)
        }
        return java.lang.Float.intBitsToFloat(bits)
    }
}
