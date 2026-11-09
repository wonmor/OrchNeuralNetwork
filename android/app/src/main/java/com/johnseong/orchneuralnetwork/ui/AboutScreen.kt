package com.johnseong.orchneuralnetwork.ui

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.johnseong.orchneuralnetwork.billing.*
import com.johnseong.orchneuralnetwork.engine.ModelConfig

/** Callbacks from the tip jar UI; the activity binds them to Play Billing. */
class TipJarActions(val onTip: (TipOption) -> Unit = {}, val onRetry: () -> Unit = {}, val onDismiss: () -> Unit = {})

@Composable
fun AboutScreen(config: ModelConfig?, parameters: Long, tips: TipJarUi, actions: TipJarActions, onClose: () -> Unit) {
    val context = LocalContext.current
    Column(Modifier.fillMaxSize().background(Palette.Paper)) {
        TopBar("About") { GlyphButton(Icons.Outlined.Close, "Close", onClose) }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp).widthIn(max = 720.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Card {
                Text("Inside LLM", style = display(24.sp))
                Footnote("See inside a language model while it predicts your next word.")
            }
            TipJarPanel(tips, actions)
            Card {
                Text("The model", style = body(17.sp, FontWeight.SemiBold))
                Footnote("DistilGPT2 by Hugging Face: a 6-layer, 82-million-parameter distillation of OpenAI's GPT-2. Weights are stored on the device as 16-bit floats and computed in 32-bit by a small native engine written for this app. The whole network is re-run from scratch on every keystroke so that every intermediate value can be shown.")
                if (config != null) Text("${config.nLayer} layers  |  ${config.nHead} heads  |  ${config.nEmbd}-dim  |  ${grouped(config.nVocab)} tokens  |  ${grouped(parameters)} parameters",
                    style = mono(11.sp, color = Palette.Muted))
                Text("Model card and Apache 2.0 licence", style = body(14.sp, color = Palette.Blue).copy(textDecoration = TextDecoration.Underline),
                    modifier = Modifier.heightIn(min = Palette.TapTarget).wrapContentHeight().clickable(role = Role.Button) {
                        runCatching { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://huggingface.co/distilgpt2"))) }
                    })
            }
            Card {
                Text("Privacy", style = body(17.sp, FontWeight.SemiBold))
                Footnote("Everything you type is processed on your device and never leaves it. No accounts, no analytics, no ads. The only network use is Google Play loading tip prices if you open the tip jar.")
            }
            Card {
                Text("A note on predictions", style = body(17.sp, FontWeight.SemiBold))
                Footnote("GPT-2 was trained on web text from 2019. Its suggestions reflect that data, can be wrong, dated or odd, and are shown for learning purposes only.")
            }
            Caption("Contact: john@orchestrsim.com")
            Spacer(Modifier.height(24.dp))
        }
    }
}

/** The tip jar panel shown in About. Prices come from Google Play; nothing here unlocks anything. */
@Composable
fun TipJarPanel(ui: TipJarUi, actions: TipJarActions) {
    Card(band = Palette.Blue) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Icon(Icons.Outlined.LocalCafe, contentDescription = null, tint = Palette.Ink, modifier = Modifier.size(24.dp))
            Text("Tip jar", style = display(20.sp), modifier = Modifier.semantics { heading() })
        }
        Text(TipJar.INTRO, style = body(15.sp))
        if (ui.status == TipStatus.Thanked) Thanks(actions) else Choices(ui, actions)
        when (val s = ui.status) {
            TipStatus.Pending -> Message(TipJar.PENDING, Icons.Outlined.HourglassEmpty, actions)
            is TipStatus.Failed -> Message(s.message, Icons.Outlined.ErrorOutline, actions)
            else -> {}
        }
        Caption(TipJar.DISCLAIMER)
    }
}

@Composable
private fun Choices(ui: TipJarUi, actions: TipJarActions) {
    when (ui.load) {
        TipLoad.Loading -> Row(Modifier.heightIn(min = Palette.TapTarget), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            CircularProgressIndicator(Modifier.size(20.dp), color = Palette.Ink, strokeWidth = 2.dp)
            Caption("Loading prices")
        }
        TipLoad.Unavailable -> Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Footnote(TipJar.UNAVAILABLE)
            ui.detail?.let { Caption(it) }
            BlockButton(actions.onRetry, Modifier.heightIn(min = Palette.TapTarget)) {
                Row(Modifier.padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Icon(Icons.Outlined.Refresh, contentDescription = null, tint = Palette.Ink, modifier = Modifier.size(18.dp))
                    Text("Try again", style = body(15.sp, FontWeight.SemiBold))
                }
            }
        }
        TipLoad.Ready -> {
            val busy = ui.status is TipStatus.Purchasing
            ui.options.forEach { o ->
                val active = ui.status == TipStatus.Purchasing(o.id)
                val fill = if (active) Palette.Blue else Palette.Sheet
                val fg = if (active) Color.White else Palette.Ink
                Row(Modifier.fillMaxWidth().heightIn(min = 52.dp).background(fill).border(Palette.Rule, Palette.Ink, RectangleShape)
                    .clickable(enabled = !busy, role = Role.Button) { actions.onTip(o) }
                    .semantics(mergeDescendants = true) { contentDescription = "${o.name}, ${o.price}. Leaves a tip; unlocks nothing." }
                    .padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(o.name.uppercase(), style = body(15.sp, FontWeight.SemiBold, fg).copy(letterSpacing = 1.2.sp), modifier = Modifier.weight(1f), maxLines = 1)
                    if (active) CircularProgressIndicator(Modifier.size(18.dp), color = fg, strokeWidth = 2.dp)
                    else Text(o.price, style = body(15.sp, FontWeight.SemiBold, fg), maxLines = 1)
                }
            }
        }
    }
}

@Composable
private fun Thanks(actions: TipJarActions) {
    Column(Modifier.fillMaxWidth().background(Palette.Paper).border(Palette.Rule, Palette.Ink, RectangleShape).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Icon(Icons.Filled.Favorite, contentDescription = null, tint = Palette.Red, modifier = Modifier.size(32.dp))
            Text("Thank you", style = display(30.sp), modifier = Modifier.semantics { heading() })
        }
        Text(TipJar.THANKS, style = body(15.sp))
        BlockButton(actions.onDismiss, Modifier.heightIn(min = Palette.TapTarget)) {
            Text("Done", style = body(15.sp, FontWeight.SemiBold), modifier = Modifier.padding(horizontal = 20.dp))
        }
    }
}

@Composable
private fun Message(text: String, icon: ImageVector, actions: TipJarActions) {
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.Top) {
        Icon(icon, contentDescription = null, tint = Palette.Red, modifier = Modifier.padding(top = 2.dp).size(20.dp))
        Text(text, style = body(14.sp), modifier = Modifier.weight(1f))
        GlyphButton(Icons.Outlined.Close, "Dismiss", actions.onDismiss)
    }
}

/** A small, quiet text link at the foot of the Steps tab. No banner, no badge, no prompt. */
@Composable
fun SupportLink(onClick: () -> Unit) {
    Row(Modifier.heightIn(min = Palette.TapTarget).clickable(role = Role.Button, onClick = onClick),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Icon(Icons.Outlined.LocalCafe, contentDescription = null, tint = Palette.Muted, modifier = Modifier.size(16.dp))
        Text("Support Inside LLM", style = body(14.sp, color = Palette.Muted).copy(textDecoration = TextDecoration.Underline))
    }
}

/** Centered Jost title on paper with a hard black rule beneath; optional trailing action. */
@Composable
fun TopBar(title: String, leading: (@Composable () -> Unit)? = null, trailing: @Composable () -> Unit = {}) {
    Column(Modifier.fillMaxWidth().background(Palette.Paper)) {
        Box(Modifier.fillMaxWidth().height(56.dp).padding(horizontal = 4.dp)) {
            if (leading != null) Box(Modifier.align(Alignment.CenterStart)) { leading() }
            Text(title, style = display(18.sp), modifier = Modifier.align(Alignment.Center).semantics { heading() }, maxLines = 1)
            Box(Modifier.align(Alignment.CenterEnd)) { trailing() }
        }
        Box(Modifier.fillMaxWidth().height(Palette.Rule).background(Palette.Ink))
    }
}
