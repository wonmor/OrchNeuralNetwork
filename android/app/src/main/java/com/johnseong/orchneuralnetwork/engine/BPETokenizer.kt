package com.johnseong.orchneuralnetwork.engine

import org.json.JSONObject
import java.util.regex.Pattern

/** GPT-2 style byte-level Byte Pair Encoding tokenizer (vocab.json + merges.txt). */
class BPETokenizer(vocabJson: String, mergesText: String) {
    private val encoder: Map<String, Int>
    private val decoder: Array<String?>
    private val bpeRanks: HashMap<String, Int> = HashMap()
    private val byteEncoder = CharArray(256)
    private val byteDecoder = HashMap<Char, Byte>()
    private val cache = HashMap<String, List<String>>()
    private val pattern = Pattern.compile(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        Pattern.UNICODE_CHARACTER_CLASS)

    val vocabularySize: Int get() = encoder.size

    init {
        val o = JSONObject(vocabJson)
        val enc = HashMap<String, Int>(o.length() * 2)
        var maxId = 0
        val keys = o.keys()
        while (keys.hasNext()) {
            val k = keys.next(); val id = o.getInt(k)
            enc[k] = id; if (id > maxId) maxId = id
        }
        encoder = enc
        decoder = arrayOfNulls(maxId + 1)
        for ((k, v) in enc) decoder[v] = k

        var rank = 0
        for (line in mergesText.split('\n')) {
            if (line.isEmpty() || line.startsWith("#")) continue
            val parts = line.split(' ')
            if (parts.size != 2) continue
            bpeRanks["${parts[0]} ${parts[1]}"] = rank++
        }

        val bs = ArrayList<Int>().apply { addAll(33..126); addAll(161..172); addAll(174..255) }
        val cs = ArrayList(bs)
        var n = 0
        for (b in 0 until 256) if (b !in bs) { bs += b; cs += 256 + n; n++ }
        for (i in bs.indices) {
            val ch = cs[i].toChar()
            byteEncoder[bs[i]] = ch
            byteDecoder[ch] = bs[i].toByte()
        }
    }

    fun encode(text: String): List<Int> {
        val ids = ArrayList<Int>()
        val m = pattern.matcher(text)
        while (m.find()) {
            val piece = m.group()
            val bytes = piece.toByteArray(Charsets.UTF_8)
            val mapped = String(CharArray(bytes.size) { byteEncoder[bytes[it].toInt() and 0xff] })
            for (token in bpe(mapped)) encoder[token]?.let { ids += it }
        }
        return ids
    }

    @Synchronized
    private fun bpe(token: String): List<String> {
        cache[token]?.let { return it }
        var word = token.map { it.toString() }
        if (word.size > 1) {
            while (true) {
                var bestRank = Int.MAX_VALUE; var bi = -1
                for (i in 0 until word.size - 1) {
                    val r = bpeRanks["${word[i]} ${word[i + 1]}"] ?: continue
                    if (r < bestRank) { bestRank = r; bi = i }
                }
                if (bi < 0) break
                val first = word[bi]; val second = word[bi + 1]
                val merged = ArrayList<String>(word.size)
                var i = 0
                while (i < word.size) {
                    if (i < word.size - 1 && word[i] == first && word[i + 1] == second) { merged += first + second; i += 2 }
                    else { merged += word[i]; i++ }
                }
                word = merged
                if (word.size == 1) break
            }
        }
        cache[token] = word
        return word
    }

    private fun bytesOf(id: Int): ByteArray {
        val t = decoder.getOrNull(id) ?: return ByteArray(0)
        val out = ByteArray(t.length); var n = 0
        for (c in t) byteDecoder[c]?.let { out[n++] = it }
        return out.copyOf(n)
    }

    /** Human-readable text for one token id (lossy for partial UTF-8, like the iOS app). */
    fun decode(id: Int): String = if (decoder.getOrNull(id) == null) "�" else String(bytesOf(id), Charsets.UTF_8)

    fun decode(ids: List<Int>): String {
        val bos = java.io.ByteArrayOutputStream()
        ids.forEach { bos.write(bytesOf(it)) }
        return String(bos.toByteArray(), Charsets.UTF_8)
    }
}
