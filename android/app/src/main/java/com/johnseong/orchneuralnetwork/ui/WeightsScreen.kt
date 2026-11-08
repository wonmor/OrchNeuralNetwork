package com.johnseong.orchneuralnetwork.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.automirrored.outlined.KeyboardArrowRight
import androidx.compose.material3.Icon
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.johnseong.orchneuralnetwork.engine.TensorInfo
import com.johnseong.orchneuralnetwork.engine.WeightFile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

/** Browse every weight matrix in the model as a heatmap. */
@Composable
fun WeightsScreen(file: WeightFile?, initialTensor: String? = null) {
    var selected by rememberSaveable { mutableStateOf(initialTensor) }
    if (file == null) {
        Column(Modifier.fillMaxSize()) { TopBar("Weights") }
        return
    }
    val sel = selected?.let { runCatching { file.info(it) }.getOrNull() }
    if (sel != null) {
        BackHandler { selected = null }
        TensorDetail(sel, file) { selected = null }
        return
    }
    val groups = remember(file) { groups(file.tensors) }
    Column(Modifier.fillMaxSize()) {
        TopBar("Weights")
        LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(0.dp)) {
            item {
                Card {
                    Text("${grouped(file.totalParameters)} parameters", style = display(22.sp))
                    Caption("${file.tensors.size} tensors  |  stored as 16-bit floats  |  ${file.model}")
                    Caption("These numbers were fixed during training. Running the model only reads them.")
                }
            }
            groups.forEach { (title, tensors) ->
                item(key = "h-$title") {
                    Text(title.uppercase(), style = display(13.sp).copy(letterSpacing = 0.8.sp, color = Palette.Muted),
                        modifier = Modifier.padding(top = 18.dp, bottom = 6.dp))
                }
                items(tensors, key = { it.name }) { t ->
                    Row(Modifier.fillMaxWidth().heightIn(min = 56.dp).background(Palette.Sheet)
                        .clickable(role = Role.Button) { selected = t.name }.padding(horizontal = 12.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(pretty(t.name), style = body(15.sp))
                            Text(t.shapeDescription, style = mono(11.sp, color = Palette.Muted))
                        }
                        Text(grouped(t.count), style = mono(11.sp, color = Palette.Muted))
                        Icon(Icons.AutoMirrored.Outlined.KeyboardArrowRight, contentDescription = null, tint = Palette.Muted)
                    }
                    Box(Modifier.fillMaxWidth().height(1.dp).background(Palette.Ink.copy(alpha = 0.12f)))
                }
            }
        }
    }
}

private fun groups(tensors: List<TensorInfo>): List<Pair<String, List<TensorInfo>>> {
    val by = LinkedHashMap<String, MutableList<TensorInfo>>()
    for (t in tensors) {
        val key = when {
            t.name.startsWith("h.") -> "Layer ${t.name.split(".")[1]}"
            t.name.startsWith("ln_f") -> "Final normalisation"
            else -> "Embeddings"
        }
        by.getOrPut(key) { mutableListOf() } += t
    }
    return by.map { it.key to it.value }
}

fun pretty(name: String): String {
    val n = if (name.startsWith("h.")) name.split(".", limit = 3)[2] else name
    return mapOf(
        "wte.weight" to "Token embeddings", "wpe.weight" to "Position embeddings",
        "ln_f.weight" to "Final LayerNorm scale", "ln_f.bias" to "Final LayerNorm shift",
        "ln_1.weight" to "LayerNorm 1 scale", "ln_1.bias" to "LayerNorm 1 shift",
        "attn.c_attn.weight" to "Attention Q, K, V projection", "attn.c_attn.bias" to "Attention Q, K, V bias",
        "attn.c_proj.weight" to "Attention output projection", "attn.c_proj.bias" to "Attention output bias",
        "ln_2.weight" to "LayerNorm 2 scale", "ln_2.bias" to "LayerNorm 2 shift",
        "mlp.c_fc.weight" to "MLP expand (768 → 3072)", "mlp.c_fc.bias" to "MLP expand bias",
        "mlp.c_proj.weight" to "MLP contract (3072 → 768)", "mlp.c_proj.bias" to "MLP contract bias",
    )[n] ?: n
}

private class TensorView(val image: ImageBitmap?, val aspect: Float, val min: Float, val max: Float, val mean: Float, val std: Float)

@Composable
private fun TensorDetail(tensor: TensorInfo, file: WeightFile, onBack: () -> Unit) {
    var view by remember(tensor.name) { mutableStateOf<TensorView?>(if (Staging.still) load(tensor, file) else null) }
    var zoom by remember(tensor.name) { mutableFloatStateOf(1f) }
    LaunchedEffect(tensor.name) { if (view == null) view = withContext(Dispatchers.Default) { load(tensor, file) } }
    Column(Modifier.fillMaxSize()) {
        TopBar("Tensor", leading = { GlyphButton(Icons.AutoMirrored.Outlined.ArrowBack, "Back to all weights", onBack) })
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Card {
                Text(pretty(tensor.name), style = body(20.sp, FontWeight.SemiBold))
                Text("${tensor.name}  |  ${tensor.shapeDescription}  |  ${grouped(tensor.count)} values", style = mono(11.sp, color = Palette.Muted))
            }
            val v = view
            if (v?.image != null) {
                BoxWithConstraints(Modifier.fillMaxWidth().height(380.dp).pointerInput(Unit) {
                    detectTransformGestures { _, _, z, _ -> zoom = (zoom * z).coerceIn(1f, 8f) }
                }) {
                    val w = maxWidth * zoom
                    Box(Modifier.fillMaxSize().horizontalScroll(rememberScrollState()).verticalScroll(rememberScrollState())) {
                        PixelImage(v.image, Modifier.width(w).height(w / v.aspect), description = "Heatmap of ${pretty(tensor.name)}")
                    }
                }
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Zoom", style = body(12.sp))
                    Slider(zoom, { zoom = it }, valueRange = 1f..8f, modifier = Modifier.weight(1f),
                        colors = SliderDefaults.colors(thumbColor = Palette.Red, activeTrackColor = Palette.Red, inactiveTrackColor = Palette.Ink.copy(alpha = 0.15f)))
                }
                Caption("Blue = negative, red = positive, pale = near zero. The grid is a sample of the full matrix; every cell is one real weight.")
                Card {
                    Row { Stat("min", v.min, Modifier.weight(1f)); Spacer(Modifier.width(16.dp)); Stat("max", v.max, Modifier.weight(1f)) }
                    Row { Stat("mean", v.mean, Modifier.weight(1f)); Spacer(Modifier.width(16.dp)); Stat("std", v.std, Modifier.weight(1f)) }
                }
            } else {
                Box(Modifier.fillMaxWidth().height(200.dp), contentAlignment = Alignment.Center) {
                    androidx.compose.material3.CircularProgressIndicator(color = Palette.Ink)
                }
            }
        }
    }
}

@Composable
private fun Stat(label: String, v: Float, modifier: Modifier) {
    Row(modifier) {
        Text(label, style = body(12.sp, color = Palette.Muted), modifier = Modifier.weight(1f))
        Text("%.4f".format(v), style = mono(12.sp))
    }
}

private fun load(t: TensorInfo, f: WeightFile): TensorView {
    val isVector = t.shape.size == 1
    val rows = if (isVector) 1 else minOf(t.shape[0], 256)
    val cols = if (isVector) minOf(t.shape[0], 768) else minOf(t.shape.drop(1).fold(1) { a, b -> a * b }, 256)
    val (values, r, c) = f.sampleGrid(t.name, rows, cols)
    val img = Heatmap.make(values, r, c, Heatmap.Mode.Diverging)
    val mean = values.average().toFloat()
    val std = sqrt(values.fold(0.0) { a, x -> a + (x - mean) * (x - mean) } / values.size).toFloat()
    // Vectors are drawn as a band, not a hairline.
    val aspect = if (r == 1) c / 24f else c.toFloat() / r
    return TensorView(img, aspect, values.min(), values.max(), mean, std)
}
