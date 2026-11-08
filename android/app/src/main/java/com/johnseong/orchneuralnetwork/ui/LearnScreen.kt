package com.johnseong.orchneuralnetwork.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.HelpOutline
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.johnseong.orchneuralnetwork.engine.Trace
import kotlin.math.abs

/** A guided walkthrough of one forward pass, using the user's own live input. */
@Composable
fun LearnScreen(trace: Trace?, parameters: Long, onSupport: () -> Unit) {
    Column(Modifier.fillMaxSize()) {
        TopBar("How it works")
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Card(band = Palette.Yellow) {
                Text("A language model is a next-word predictor", style = display(22.sp))
                Footnote("Your phone keyboard, chatbots and this app all do the same core thing: turn text into numbers, push the numbers through a stack of layers, and read off a probability for every possible next token. Then they do it again for the next word. Here is one pass, step by step, on your own text.")
            }
            if (trace != null) {
                Step(1, "Tokenize", "Text is chopped into tokens, pieces of words from a fixed list of 50,257. Common words are one token, rare ones are several. Your input became ${trace.tokenCount} tokens. ␣ marks a leading space, which is part of the token.") {
                    TokenStrip(trace.tokens)
                }
                Step(2, "Embed", "Each token id looks up a row in a 50,257 × 768 table, giving a vector of 768 numbers. A second table adds the token's position. Shown: the first 64 numbers of the last token's vector. The model never sees letters again, only these vectors.") {
                    Row(Modifier.fillMaxWidth().height(44.dp), horizontalArrangement = Arrangement.spacedBy(1.dp), verticalAlignment = Alignment.CenterVertically) {
                        trace.embeddingPreview.forEach { v ->
                            Box(Modifier.weight(1f).height((minOf(abs(v) * 60, 40f) + 1).dp).background(if (v >= 0) Palette.Yellow else Palette.Blue))
                        }
                    }
                }
                val T = trace.tokenCount
                val l0 = trace.layers[0]
                val avg = FloatArray(T * T).also { a -> for (h in l0.attention) for (i in 0 until T * T) a[i] += h[i] / l0.attention.size }
                Step(3, "Attention: tokens look at each other", "In every layer, each token computes a query, and every earlier token offers a key. Where they match, information flows. Twelve heads do this in parallel, each learning its own habit: one tracks the previous word, another finds the subject of the sentence. This is layer 0, averaged over heads.") {
                    AttentionHeatmap(avg, trace.tokens, minOf(260, T * 20 + 40).dp)
                }
                val last = trace.layers.last()
                Step(4, "MLP: each token thinks on its own", "After attention, each token's vector is expanded to 3,072 numbers, passed through a nonlinearity, and squeezed back to 768. This is where much of the model's stored knowledge lives. In the last layer ${(last.mlpActiveFraction * 100).toInt()}% of neurons fired for your last token.") {
                    NeuronGrid(last.mlpActivations, 100.dp)
                }
                Step(5, "Six layers, one refined guess", "Attention and MLP repeat six times. The 'logit lens' reads a prediction after every layer, so you can see the answer form. Early layers often make a crude guess; later layers refine or overturn it.") {
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        trace.layers.forEach { l ->
                            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                Text("Layer ${l.id}", style = mono(11.sp), modifier = Modifier.width(60.dp))
                                Row(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                    l.logitLens.take(3).forEach { p -> TokenChip(p.text, p.prob) }
                                }
                                Text(l.logitLens.firstOrNull()?.let { percent(it.prob) } ?: "", style = mono(11.sp, color = Palette.Muted))
                            }
                        }
                    }
                }
                Step(6, "Read off probabilities", "The final vector is compared with all 50,257 token embeddings to score each one, and a softmax turns the scores into probabilities. A keyboard just shows the top three words. A chatbot samples one, appends it, and runs the whole thing again.") {
                    ProbabilityBars(trace.topK, limit = 5)
                }
            } else {
                Card { Footnote("Type something in the Predict tab and come back: every step below will use your words.") }
            }
            Card {
                SectionTitle("Why can't I see every weight at once?", icon = Icons.Outlined.HelpOutline, color = Palette.Blue)
                Footnote("This small model has ${grouped(parameters)} weights. A 4K screen has about 8 million pixels. Frontier models have hundreds of billions of weights, so any picture of a network is a sample. The Weights tab lets you page through the real matrices, one at a time.")
            }
            SupportLink(onSupport)
        }
    }
}

@Composable
private fun Step(n: Int, title: String, text: String, content: @Composable () -> Unit) {
    val fill: Color = listOf(Palette.Red, Palette.Blue, Palette.Yellow)[(n - 1) % 3]
    Card {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("$n", style = display(28.sp).copy(color = if (fill == Palette.Yellow) Palette.Ink else fill))
            Text(title.uppercase(), style = display(15.sp).copy(letterSpacing = 0.5.sp), modifier = Modifier.semantics { heading() })
        }
        content()
        Footnote(text)
    }
}
