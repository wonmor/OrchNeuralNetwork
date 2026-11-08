package com.johnseong.orchneuralnetwork.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.johnseong.orchneuralnetwork.engine.InferenceEngine
import com.johnseong.orchneuralnetwork.engine.LayerTrace
import com.johnseong.orchneuralnetwork.engine.Trace
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** Layer-by-layer visualiser: attention heatmaps, neuron activity, and the logit lens. */
@Composable
fun InsideScreen(engine: InferenceEngine, initialLayer: Int = 0, initialHead: Int? = null) {
    var layer by rememberSaveable { mutableIntStateOf(initialLayer) }
    var head by rememberSaveable { mutableStateOf(initialHead) } // null = average of all heads
    var playhead by remember { mutableIntStateOf(5) }
    var playing by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val trace = engine.trace

    suspend fun replay() {
        if (engine.trace == null || Staging.still) return
        playing = true; playhead = -1
        for (i in 0 until 6) { delay(220); playhead = i }
        playing = false
    }
    LaunchedEffect(trace?.serial) { replay() }

    Column(Modifier.fillMaxSize()) {
        TopBar("Inside the network")
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            SentenceField(engine.text, engine::updateText, "Type a sentence…", 1, 4, big = false)
            if (trace != null) {
                LayerStack(trace, layer, playhead, playing, onSelect = { layer = it }, onReplay = { scope.launch { replay() } })
                AttentionCard(trace, trace.layers[layer.coerceIn(0, trace.layers.size - 1)], head) { head = it }
                NeuronsCard(trace.layers[layer.coerceIn(0, trace.layers.size - 1)])
            } else {
                Footnote("Type above to run the network.", Modifier.fillMaxWidth().padding(top = 40.dp))
            }
        }
    }
}

@Composable
private fun LayerStack(trace: Trace, layer: Int, playhead: Int, playing: Boolean, onSelect: (Int) -> Unit, onReplay: () -> Unit) {
    Card {
        Row(verticalAlignment = Alignment.CenterVertically) {
            SectionTitle("The 6 layers, in order", "Tap a layer to inspect it  |  bar = size of the residual vector",
                Icons.Outlined.Layers, Palette.Blue, Modifier.weight(1f))
            GlyphButton(Icons.Outlined.PlayCircle, "Replay the forward pass", onReplay, Palette.Red, enabled = !playing)
        }
        TokenStrip(trace.tokens, highlightLast = true)
        Caption("Embedding  |  ‖x‖ = ${"%.1f".format(trace.embeddingNorm)}")
        val maxNorm = trace.layers.maxOf { it.residualNorm }.coerceAtLeast(1e-6f)
        Column {
            trace.layers.forEach { l ->
                val shown = l.id <= playhead
                Row(Modifier.fillMaxWidth().heightIn(min = 48.dp)
                    .background(if (layer == l.id) Palette.Yellow.copy(alpha = 0.45f) else Palette.Sheet)
                    .clickable(role = Role.Button) { onSelect(l.id) }.padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("L${l.id}", style = mono(12.sp, FontWeight.Bold), modifier = Modifier.width(26.dp))
                    Box(Modifier.weight(1f).height(10.dp).background(Palette.Ink.copy(alpha = 0.06f))) {
                        if (shown) Box(Modifier.fillMaxHeight().fillMaxWidth((l.residualNorm / maxNorm).coerceIn(0.03f, 1f))
                            .background(if (l.id == layer) Palette.Red else Palette.Blue))
                    }
                    Column(Modifier.width(140.dp), horizontalAlignment = Alignment.End) {
                        val top = l.logitLens.firstOrNull()
                        if (shown && top != null) {
                            Text("→ ${top.text.trim()}", style = body(12.sp, FontWeight.SemiBold), maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text("${percent(top.prob)}  |  ${(l.mlpActiveFraction * 100).toInt()}% neurons on", style = body(10.sp, color = Palette.Muted), maxLines = 1)
                        } else Caption("…")
                    }
                }
            }
        }
        Caption("The arrow shows what the model would predict if it stopped after that layer (the “logit lens”). Watch the guess sharpen or change as the signal goes deeper.")
    }
}

private fun attentionMatrix(l: LayerTrace, T: Int, head: Int?): FloatArray {
    if (head != null) return l.attention[head]
    val avg = FloatArray(T * T)
    for (h in l.attention) for (i in 0 until T * T) avg[i] += h[i]
    val inv = 1f / l.attention.size
    for (i in avg.indices) avg[i] *= inv
    return avg
}

@Composable
private fun AttentionCard(trace: Trace, l: LayerTrace, head: Int?, onHead: (Int?) -> Unit) {
    val T = trace.tokenCount
    val matrix = remember(trace.serial, l.id, head) { attentionMatrix(l, T, head) }
    val lastRow = matrix.copyOfRange((T - 1) * T, T * T)
    var open by remember { mutableStateOf(false) }
    Card {
        SectionTitle("Attention  |  layer ${l.id}", "Each row is a token looking back at earlier tokens. Darker = more weight.",
            Icons.Outlined.Visibility, Palette.Red)
        Box {
            BlockButton({ open = true }, Modifier.heightIn(min = 44.dp)) {
                Row(Modifier.padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(if (head == null) "Average of 12 heads" else "Head $head", style = body(14.sp, FontWeight.SemiBold, Palette.Blue))
                    Icon(Icons.Outlined.ExpandMore, contentDescription = "Choose head", tint = Palette.Blue, modifier = Modifier.size(18.dp))
                }
            }
            DropdownMenu(open, { open = false }) {
                DropdownMenuItem({ Text("Average", style = body(15.sp)) }, { onHead(null); open = false })
                for (h in 0 until 12) DropdownMenuItem({ Text("Head $h", style = body(15.sp)) }, { onHead(h); open = false })
            }
        }
        Caption("What the last token “${trace.tokens.last().trim()}” is reading from:")
        TokenStrip(trace.tokens, lastRow, highlightLast = true)
        AttentionHeatmap(matrix, trace.tokens, minOf(340, T * 22 + 60).dp)
    }
}

@Composable
private fun NeuronsCard(l: LayerTrace) {
    Card {
        SectionTitle("MLP neurons  |  layer ${l.id}", "3,072 neurons for the last token  |  ${(l.mlpActiveFraction * 100).toInt()}% fired",
            Icons.Outlined.GridOn, Palette.Yellow)
        NeuronGrid(l.mlpActivations, 120.dp)
        Caption("After attention mixes information between tokens, each token passes through this block alone. These are the raw GELU outputs: most neurons stay quiet, a few fire hard.")
    }
}
