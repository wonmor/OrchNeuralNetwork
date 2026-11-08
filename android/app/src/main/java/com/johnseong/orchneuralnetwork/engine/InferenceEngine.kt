package com.johnseong.orchneuralnetwork.engine

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class Suggestion(
    /** The full word to show in the suggestion bar. */
    val display: String,
    /** The exact text to append to the input when tapped. */
    val insertion: String,
    val prob: Float,
)

sealed interface EngineState {
    data object Loading : EngineState
    data object Ready : EngineState
    data class Failed(val message: String) : EngineState
}

/** Owns the model, debounces input, and publishes the latest trace to the UI. */
class InferenceEngine {
    var state: EngineState by mutableStateOf(EngineState.Loading); private set
    var trace: Trace? by mutableStateOf(null); private set
    var suggestions: List<Suggestion> by mutableStateOf(emptyList()); private set
    var isRunning by mutableStateOf(false); private set
    var model: GPT2Model? by mutableStateOf(null); private set
    var text: String by mutableStateOf("")
        private set

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var pending: Job? = null
    private var generation = 0

    val parameterCount: Long get() = model?.file?.totalParameters ?: 0
    val config: ModelConfig? get() = model?.config

    fun updateText(value: String) {
        if (value == text) return
        text = value
        schedule()
    }

    fun load(context: Context, prefill: String? = null) {
        if (model != null) return
        if (text.isEmpty() && !prefill.isNullOrEmpty()) text = prefill
        val app = context.applicationContext
        scope.launch {
            try {
                val m = withContext(Dispatchers.Default) { loadBundled(app) }
                model = m
                state = EngineState.Ready
                schedule()
            } catch (e: Throwable) {
                state = EngineState.Failed(e.message ?: e.toString())
            }
        }
    }

    /** Runs synchronously with a finished model; used to stage store screenshots on the JVM. */
    fun stage(m: GPT2Model, input: String) {
        model = m; state = EngineState.Ready; text = input
        trace = m.run(input)
        suggestions = trace?.let { makeSuggestions(it, input) } ?: emptyList()
    }

    fun run(now: Boolean = false) = schedule(if (now) 0 else 250)

    private fun schedule(delayMs: Long = 250) {
        val m = model ?: return
        pending?.cancel()
        val gen = ++generation
        val input = text
        pending = scope.launch {
            if (delayMs > 0) delay(delayMs)
            isRunning = true
            val result = withContext(Dispatchers.Default) { m.run(input) }
            if (gen != generation) return@launch
            isRunning = false
            trace = result
            suggestions = if (result != null) makeSuggestions(result, input) else emptyList()
        }
    }

    fun accept(s: Suggestion) {
        text += s.insertion
        run(now = true)
    }

    companion object {
        private val blocked = setOf("fuck", "shit", "cunt", "nigger", "faggot", "bitch", "asshole", "dick", "pussy", "rape")

        fun loadBundled(context: Context): GPT2Model {
            val am = context.assets
            val file = am.openFd("distilgpt2.tsw").use { fd -> WeightFile.map(fd.fileDescriptor, fd.startOffset, fd.length) }
            val vocab = am.open("vocab.json").bufferedReader().use { it.readText() }
            val merges = am.open("merges.txt").bufferedReader().use { it.readText() }
            return GPT2Model(file, BPETokenizer(vocab, merges), if (NativeKernels.available) NativeKernels else JvmKernels())
        }

        private fun isWordChar(c: Char) = Character.isLetter(c) || c == '\''

        /** What a keyboard would do with the raw distribution: three whole words, no duplicates. */
        fun makeSuggestions(trace: Trace, input: String): List<Suggestion> {
            val endsWithSpace = input.lastOrNull()?.isWhitespace() ?: true
            val lastWord = if (endsWithSpace) "" else input.split(Regex("\\s+")).lastOrNull { it.isNotEmpty() } ?: ""
            val seen = HashSet<String>()
            val out = ArrayList<Suggestion>()
            for (cand in trace.topK) {
                val raw = cand.text
                val trimmed = raw.trim()
                if (trimmed.isEmpty() || !trimmed.all(::isWordChar)) continue
                val startsWord = raw.firstOrNull() == ' '
                val display: String; val insertion: String
                if (startsWord) {
                    display = trimmed
                    insertion = if (endsWithSpace) "$trimmed " else "$raw "
                } else {
                    if (endsWithSpace) continue // a continuation makes no sense after a space
                    display = lastWord + trimmed
                    insertion = "$trimmed "
                }
                val key = display.lowercase()
                if (key in seen || key in blocked) continue
                seen += key
                out += Suggestion(display, insertion, cand.prob)
                if (out.size == 3) break
            }
            return out
        }
    }
}
