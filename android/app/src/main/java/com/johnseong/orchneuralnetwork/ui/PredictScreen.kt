package com.johnseong.orchneuralnetwork.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowForward
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.johnseong.orchneuralnetwork.engine.InferenceEngine
import com.johnseong.orchneuralnetwork.engine.Trace
import kotlinx.coroutines.delay

/** The keyboard demo: type, get suggestions, and watch the network fire. */
@Composable
fun PredictScreen(engine: InferenceEngine, onAbout: () -> Unit, onOpenInside: () -> Unit) {
    Column(Modifier.fillMaxSize()) {
        TopBar("Inside LLM") { GlyphButton(Icons.Outlined.Info, "About and tip jar", onAbout) }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Banner()
            InputCard(engine)
            SuggestionBar(engine)
            PulseCard(engine.trace, onOpenInside)
            engine.trace?.let { trace ->
                Card {
                    SectionTitle("What the model thinks comes next", "Probability over all 50,257 tokens, top 10 shown",
                        Icons.Outlined.BarChart, Palette.Yellow)
                    ProbabilityBars(trace.topK)
                }
                Card {
                    SectionTitle("Your text as tokens", "${trace.tokenCount} tokens  |  ␣ marks a leading space", Icons.Outlined.DataArray, Palette.Red)
                    TokenStrip(trace.tokens, highlightLast = true)
                }
            }
        }
    }
}

/** Header: a glyph for the network and the app's promise. */
@Composable
private fun Banner() {
    Row(Modifier.fillMaxWidth().background(Palette.Sheet).border(Palette.Rule, Palette.Ink, RectangleShape)
        .semantics(mergeDescendants = true) { contentDescription = "Watch a language model think. 82 million weights, 6 layers, on this device." },
        verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.width(8.dp).height(84.dp).background(Palette.Red))
        Box(Modifier.padding(horizontal = 14.dp).size(56.dp).background(Palette.Yellow).border(Palette.Rule, Palette.Ink, RectangleShape),
            contentAlignment = Alignment.Center) {
            Icon(Icons.Outlined.Hub, contentDescription = null, tint = Palette.Ink, modifier = Modifier.size(36.dp))
        }
        Column(Modifier.weight(1f).padding(end = 12.dp, top = 10.dp, bottom = 10.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("WATCH A LANGUAGE MODEL THINK", style = display(17.sp), maxLines = 2)
            Caption("82 million weights  |  6 layers  |  on this device")
        }
    }
}

@Composable
fun SentenceField(value: String, onChange: (String) -> Unit, placeholder: String, minLines: Int, maxLines: Int, big: Boolean) {
    // Local selection state; text set from outside (a suggestion or sample chip) puts the cursor at the end.
    var field by remember { mutableStateOf(TextFieldValue(value, TextRange(value.length))) }
    if (field.text != value) field = TextFieldValue(value, TextRange(value.length))
    BasicTextField(field, { field = it; onChange(it.text) },
        textStyle = body(if (big) 20.sp else 16.sp),
        cursorBrush = SolidColor(Palette.Red),
        minLines = minLines, maxLines = maxLines,
        keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.None, autoCorrectEnabled = false),
        modifier = Modifier.fillMaxWidth().background(Palette.Sheet).border(Palette.Rule, Palette.Ink, RectangleShape).padding(12.dp),
        decorationBox = { inner ->
            Box {
                if (value.isEmpty()) Text(placeholder, style = body(if (big) 20.sp else 16.sp, color = Palette.Muted))
                inner()
            }
        })
}

@Composable
private fun InputCard(engine: InferenceEngine) {
    Card {
        Row(verticalAlignment = Alignment.CenterVertically) {
            SectionTitle("Type something", "A 6-layer GPT-2 predicts the next word live", Icons.Outlined.Keyboard, Palette.Blue, Modifier.weight(1f))
            if (engine.isRunning) CircularProgressIndicator(Modifier.size(16.dp), color = Palette.Ink, strokeWidth = 2.dp)
        }
        SentenceField(engine.text, engine::updateText, "I am going to the", 2, 6, big = true)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf("I am going to the", "The weather today is", "My favourite food is").forEach { s ->
                BlockButton({ engine.updateText("$s ") }, Modifier.weight(1f).heightIn(min = 40.dp)) {
                    Text(s.split(" ").take(2).joinToString(" ") + "…", style = body(12.sp, FontWeight.SemiBold), maxLines = 1,
                        overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(horizontal = 6.dp, vertical = 6.dp))
                }
            }
        }
    }
}

@Composable
private fun SuggestionBar(engine: InferenceEngine) {
    val haptics = LocalHapticFeedback.current
    Card {
        SectionTitle("Suggestions", "Exactly what a keyboard shows: the three most probable next words", Icons.Outlined.AutoAwesome, Palette.Red)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (engine.suggestions.isEmpty()) {
                Box(Modifier.fillMaxWidth().heightIn(min = 52.dp), contentAlignment = Alignment.Center) {
                    Footnote(if (engine.text.isEmpty()) "Start typing…" else "No word-like suggestions")
                }
            } else {
                engine.suggestions.forEachIndexed { idx, s ->
                    BlockButton({ engine.accept(s); haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove) },
                        Modifier.weight(1f).heightIn(min = 56.dp), fill = if (idx == 0) Palette.Yellow else Palette.Sheet) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(4.dp)) {
                            Text(s.display, style = body(16.sp, FontWeight.Medium), maxLines = 1, overflow = TextOverflow.Ellipsis, textAlign = TextAlign.Center)
                            Text(percent(s.prob), style = body(11.sp, color = Palette.Muted))
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PulseCard(trace: Trace?, onOpenInside: () -> Unit) {
    Card {
        Row(verticalAlignment = Alignment.CenterVertically) {
            SectionTitle("Signal through the network",
                trace?.let { "6 layers  |  ${it.elapsedMilliseconds.toInt()} ms on device" } ?: "Waiting for input",
                Icons.Outlined.Layers, Palette.Blue, Modifier.weight(1f))
            BlockButton(onOpenInside, Modifier.heightIn(min = 40.dp)) {
                Row(Modifier.padding(horizontal = 10.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Open", style = body(12.sp, FontWeight.SemiBold))
                    Icon(Icons.AutoMirrored.Outlined.ArrowForward, contentDescription = null, modifier = Modifier.size(14.dp))
                }
            }
        }
        LayerPulse(trace)
    }
}

/** Six bars that light up in sequence every time a forward pass completes. */
@Composable
fun LayerPulse(trace: Trace?) {
    var lit by remember { mutableIntStateOf(if (trace != null) 5 else -1) }
    LaunchedEffect(trace?.serial) {
        if (trace == null) { lit = -1; return@LaunchedEffect }
        if (Staging.still) { lit = 5; return@LaunchedEffect }
        lit = -1
        for (i in 0 until 6) { delay(90); lit = i }
    }
    val maxN = trace?.layers?.maxOfOrNull { it.residualNorm }?.coerceAtLeast(1e-6f) ?: 1f
    Row(Modifier.fillMaxWidth().height(70.dp), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.Bottom) {
        for (i in 0 until 6) {
            val frac = trace?.layers?.getOrNull(i)?.residualNorm?.div(maxN) ?: 0f
            val h by animateFloatAsState(28f + frac * 26f, tween(120), label = "bar")
            Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Box(Modifier.fillMaxWidth().height(h.dp).background(when {
                    i == lit -> Palette.Red; i < lit -> Palette.Blue; else -> Palette.Ink.copy(alpha = 0.08f)
                }))
                Text("L$i", style = body(10.sp, color = Palette.Muted))
            }
        }
    }
}
