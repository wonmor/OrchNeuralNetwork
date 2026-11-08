package com.johnseong.orchneuralnetwork.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.johnseong.orchneuralnetwork.R
import com.johnseong.orchneuralnetwork.engine.TokenProb
import java.text.NumberFormat
import kotlin.math.abs
import kotlin.math.sqrt

/** Bauhaus palette shared with the iOS app: three primaries and black on warm paper. */
object Palette {
    val Paper = Color(0xFFF2ECE0)
    val Sheet = Color(0xFFFCF9F2)
    val Ink = Color(0xFF141414)
    val Red = Color(0xFFD52B1E)
    val Yellow = Color(0xFFF4BE00)
    val Blue = Color(0xFF1740A0)
    val Muted = Color(0xFF141414).copy(alpha = 0.62f)
    val Rule = 2.dp
    val TapTarget = 48.dp

    private val stops = arrayOf(floatArrayOf(0.949f, 0.925f, 0.878f), floatArrayOf(0.957f, 0.745f, 0f),
        floatArrayOf(0.835f, 0.169f, 0.118f), floatArrayOf(0.078f, 0.078f, 0.078f))

    /** Sequential ramp for attention and activations: paper, yellow, red, black. */
    fun heatRGB(v: Float): FloatArray {
        val t = v.coerceIn(0f, 1f)
        val x = t * 3; val i = minOf(x.toInt(), 2); val k = x - i
        val a = stops[i]; val b = stops[i + 1]
        return FloatArray(3) { a[it] + (b[it] - a[it]) * k }
    }

    fun heat(v: Float): Color = heatRGB(v).let { Color(it[0], it[1], it[2]) }
}

@OptIn(androidx.compose.ui.text.ExperimentalTextApi::class)
private fun jost(weight: Int) = Font(R.font.jost, FontWeight(weight), variationSettings = FontVariation.Settings(FontVariation.weight(weight)))

/** Jost (SIL Open Font License), the geometric face standing in for Futura on Android. */
val Jost = FontFamily(jost(400), jost(500), jost(600), jost(700))

fun body(size: TextUnit = 15.sp, weight: FontWeight = FontWeight.Normal, color: Color = Palette.Ink) =
    TextStyle(fontFamily = Jost, fontSize = size, fontWeight = weight, color = color, lineHeight = size * 1.35f)

fun display(size: TextUnit) = TextStyle(fontFamily = Jost, fontSize = size, fontWeight = FontWeight.Bold, color = Palette.Ink,
    lineHeight = size * 1.15f)

fun mono(size: TextUnit = 12.sp, weight: FontWeight = FontWeight.Normal, color: Color = Palette.Ink) =
    TextStyle(fontFamily = FontFamily.Monospace, fontSize = size, fontWeight = weight, color = color)

/** Flat sheet with a hard black rule and square corners; optional colour band across the top. */
@Composable
fun Card(band: Color? = null, modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Column(modifier.fillMaxWidth().background(Palette.Sheet).border(Palette.Rule, Palette.Ink, RectangleShape)) {
        if (band != null) Box(Modifier.fillMaxWidth().height(6.dp).background(band))
        Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp), content = content)
    }
}

/** Uppercase section heading led by a coloured glyph. */
@Composable
fun SectionTitle(text: String, subtitle: String? = null, icon: ImageVector, color: Color = Palette.Red, modifier: Modifier = Modifier) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(18.dp))
            Text(text.uppercase(), style = display(15.sp).copy(letterSpacing = 0.6.sp), modifier = Modifier.semantics { heading() })
        }
        if (subtitle != null) Text(subtitle, style = body(12.sp, color = Palette.Muted))
    }
}

@Composable
fun Caption(text: String, modifier: Modifier = Modifier) = Text(text, style = body(12.sp, color = Palette.Muted), modifier = modifier)

@Composable
fun Footnote(text: String, modifier: Modifier = Modifier) = Text(text, style = body(14.sp, color = Palette.Muted), modifier = modifier)

/** Ruled block button used for chips and suggestions. */
@Composable
fun BlockButton(onClick: () -> Unit, modifier: Modifier = Modifier, fill: Color = Palette.Sheet, enabled: Boolean = true,
                content: @Composable BoxScope.() -> Unit) {
    Box(modifier.background(fill).border(Palette.Rule, Palette.Ink, RectangleShape)
        .clickable(enabled = enabled, role = Role.Button, onClick = onClick), contentAlignment = Alignment.Center, content = content)
}

@Composable
fun GlyphButton(icon: ImageVector, label: String, onClick: () -> Unit, tint: Color = Palette.Ink, enabled: Boolean = true) {
    Box(Modifier.size(Palette.TapTarget).clickable(enabled = enabled, role = Role.Button, onClick = onClick), contentAlignment = Alignment.Center) {
        Icon(icon, contentDescription = label, tint = if (enabled) tint else Palette.Muted)
    }
}

private fun chipText(text: String): String {
    val s = text.replace("\n", "⏎")
    return when { s.startsWith(" ") -> "␣" + s.drop(1); s.isEmpty() -> "∅"; else -> s }
}

@Composable
fun TokenChip(text: String, weight: Float = 0f, highlighted: Boolean = false) {
    Text(chipText(text), style = mono(13.sp, color = if (weight > 0.55f) Color.White else Palette.Ink), maxLines = 1,
        modifier = Modifier.background(if (weight > 0.02f) Palette.heat(weight) else Palette.Sheet)
            .border(if (highlighted) 3.dp else 1.5.dp, if (highlighted) Palette.Red else Palette.Ink, RectangleShape)
            .padding(horizontal = 8.dp, vertical = 5.dp))
}

@Composable
fun TokenStrip(tokens: List<String>, weights: FloatArray? = null, highlightLast: Boolean = false) {
    Row(Modifier.horizontalScroll(rememberScrollState()).padding(vertical = 2.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        tokens.forEachIndexed { i, t -> TokenChip(t, weights?.getOrNull(i) ?: 0f, highlightLast && i == tokens.size - 1) }
    }
}

@Composable
fun ProbabilityBars(items: List<TokenProb>, limit: Int = 10) {
    val shown = items.take(limit)
    val maxP = maxOf(shown.firstOrNull()?.prob ?: 1f, 1e-6f)
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        shown.forEachIndexed { i, item ->
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Box(Modifier.width(96.dp)) { TokenChip(item.text) }
                Box(Modifier.weight(1f).height(12.dp).background(Palette.Ink.copy(alpha = 0.06f))) {
                    Box(Modifier.fillMaxHeight().fillMaxWidth((item.prob / maxP).coerceIn(0.02f, 1f))
                        .background(if (i == 0) Palette.Red else Palette.Blue))
                }
                Text(percent(item.prob), style = mono(12.sp, color = Palette.Muted), modifier = Modifier.width(56.dp),
                    textAlign = androidx.compose.ui.text.style.TextAlign.End)
            }
        }
    }
}

fun percent(p: Float): String = when {
    p >= 0.1f -> "%.0f%%".format(p * 100)
    p >= 0.01f -> "%.1f%%".format(p * 100)
    else -> "%.2f%%".format(p * 100)
}

fun grouped(n: Long): String = NumberFormat.getIntegerInstance().format(n)
fun grouped(n: Int): String = grouped(n.toLong())

/** Small raster heatmaps from float arrays; far cheaper than thousands of composables. */
object Heatmap {
    enum class Mode { Sequential, Diverging }

    fun make(values: FloatArray, rows: Int, cols: Int, mode: Mode, scale: Float? = null): ImageBitmap? {
        if (rows <= 0 || cols <= 0 || values.size < rows * cols) return null
        val n = rows * cols
        val s = scale ?: when (mode) {
            Mode.Sequential -> maxOf(values.take(n).maxOrNull() ?: 1f, 1e-6f)
            Mode.Diverging -> {
                var mean = 0.0; for (i in 0 until n) mean += values[i]; mean /= n
                var v = 0.0; for (i in 0 until n) v += (values[i] - mean) * (values[i] - mean)
                maxOf(2.5f * sqrt((v / n).toFloat()), 1e-6f)
            }
        }
        val px = IntArray(n)
        val p = floatArrayOf(0.949f, 0.925f, 0.878f)
        for (i in 0 until n) {
            val v = values[i] / s
            val rgb = when (mode) {
                Mode.Sequential -> Palette.heatRGB(v)
                Mode.Diverging -> {
                    val t = v.coerceIn(-1f, 1f)
                    val e = if (t < 0) floatArrayOf(0.090f, 0.251f, 0.627f) else floatArrayOf(0.835f, 0.169f, 0.118f)
                    val k = abs(t)
                    FloatArray(3) { p[it] + (e[it] - p[it]) * k }
                }
            }
            px[i] = (0xff shl 24) or ((rgb[0] * 255).toInt().coerceIn(0, 255) shl 16) or
                ((rgb[1] * 255).toInt().coerceIn(0, 255) shl 8) or (rgb[2] * 255).toInt().coerceIn(0, 255)
        }
        return Bitmap.createBitmap(px, cols, rows, Bitmap.Config.ARGB_8888).asImageBitmap()
    }
}

/** A heatmap drawn with hard pixel edges inside a thin black frame. */
@Composable
fun PixelImage(bitmap: ImageBitmap, modifier: Modifier = Modifier, contentScale: ContentScale = ContentScale.FillBounds, description: String? = null) {
    Image(bitmap, contentDescription = description, modifier = modifier.border(1.dp, Palette.Ink, RectangleShape),
        contentScale = contentScale, filterQuality = FilterQuality.None)
}

@Composable
fun AttentionHeatmap(matrix: FloatArray, tokens: List<String>, height: Dp) {
    val T = tokens.size
    val image = remember(matrix) { Heatmap.make(matrix, T, T, Heatmap.Mode.Sequential, 1f) }
    BoxWithConstraints(Modifier.fillMaxWidth().height(height)) {
        val labelW = 54.dp
        val side = minOf(maxWidth - labelW, maxHeight - 16.dp)
        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            Column {
                tokens.forEach { t ->
                    Box(Modifier.width(labelW - 4.dp).height(side / T), contentAlignment = Alignment.CenterEnd) {
                        Text(t.trim(), style = mono(8.sp), maxLines = 1, overflow = TextOverflow.Clip)
                    }
                }
            }
            if (image != null) PixelImage(image, Modifier.size(side), description = "Attention heatmap")
        }
    }
}

@Composable
fun NeuronGrid(activations: FloatArray, height: Dp) {
    val image = remember(activations) {
        val cols = 96; val rows = activations.size / cols
        Heatmap.make(FloatArray(activations.size) { maxOf(activations[it], 0f) }, rows, cols, Heatmap.Mode.Sequential, 4f)
    }
    if (image != null) PixelImage(image, Modifier.fillMaxWidth().height(height), description = "MLP neuron activations")
}

/** Set by the JVM screenshot renderer so playback animations show their final frame. */
object Staging { var still = false }
