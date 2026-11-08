package com.johnseong.orchneuralnetwork

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.FormatListNumbered
import androidx.compose.material.icons.outlined.ErrorOutline
import androidx.compose.material.icons.outlined.GridOn
import androidx.compose.material.icons.outlined.Keyboard
import androidx.compose.material.icons.outlined.Psychology
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.johnseong.orchneuralnetwork.billing.TipJarBilling
import com.johnseong.orchneuralnetwork.billing.TipJarUi
import com.johnseong.orchneuralnetwork.engine.EngineState
import com.johnseong.orchneuralnetwork.engine.InferenceEngine
import com.johnseong.orchneuralnetwork.ui.*

class MainActivity : ComponentActivity() {
    private val engine = InferenceEngine()
    private lateinit var billing: TipJarBilling

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        billing = TipJarBilling(this)
        // `adb shell am start -n ... --es prefill "text" --ei tab N --ez showAbout true` (QA and screenshots).
        engine.load(this, intent.getStringExtra("prefill"))
        val startTab = intent.getIntExtra("tab", 0)
        val startAbout = intent.getBooleanExtra("showAbout", false)
        setContent {
            var tab by rememberSaveable { mutableIntStateOf(startTab) }
            var about by rememberSaveable { mutableStateOf(startAbout) }
            LaunchedEffect(about) { if (about) billing.start() }
            val actions = TipJarActions(onTip = { billing.tip(this, it) }, onRetry = billing::retry, onDismiss = billing::dismiss)
            AppShell(engine, tab, { tab = it }, about, { about = it }, billing.ui, actions)
        }
    }

    override fun onDestroy() {
        billing.close()
        super.onDestroy()
    }
}

private class TabSpec(val title: String, val icon: ImageVector)

private val tabs = listOf(TabSpec("Predict", Icons.Outlined.Keyboard), TabSpec("Inside", Icons.Outlined.Psychology),
    TabSpec("Weights", Icons.Outlined.GridOn), TabSpec("Steps", Icons.Outlined.FormatListNumbered))

/** Four tabs like the iOS app, About (with the tip jar) as a full-screen layer, and the loading state. */
@Composable
fun AppShell(engine: InferenceEngine, tab: Int, onTab: (Int) -> Unit, about: Boolean, onAbout: (Boolean) -> Unit,
             tips: TipJarUi, actions: TipJarActions, insideLayer: Int = 0, insideHead: Int? = null, weightsTensor: String? = null) {
    MaterialTheme(colorScheme = lightColorScheme(primary = Palette.Red, background = Palette.Paper, surface = Palette.Paper,
        onSurface = Palette.Ink, surfaceContainer = Palette.Sheet)) {
        Box(Modifier.fillMaxSize().background(Palette.Paper).windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Horizontal))) {
            Column(Modifier.fillMaxSize()) {
                Box(Modifier.weight(1f).fillMaxWidth().windowInsetsPadding(WindowInsets.statusBars).consumeWindowInsets(WindowInsets.navigationBars)
                    .imePadding()) {
                    when (tab) {
                        0 -> PredictScreen(engine, onAbout = { onAbout(true) }, onOpenInside = { onTab(1) })
                        1 -> InsideScreen(engine, insideLayer, insideHead)
                        2 -> WeightsScreen(engine.model?.file, weightsTensor)
                        else -> LearnScreen(engine.trace, engine.parameterCount, onSupport = { onAbout(true) })
                    }
                }
                Box(Modifier.fillMaxWidth().height(Palette.Rule).background(Palette.Ink))
                NavigationBar(containerColor = Palette.Paper, tonalElevation = 0.dp) {
                    tabs.forEachIndexed { i, t ->
                        NavigationBarItem(selected = tab == i, onClick = { onTab(i) },
                            icon = { Icon(t.icon, contentDescription = null) },
                            label = { Text(t.title, style = body(12.sp)) },
                            colors = NavigationBarItemDefaults.colors(selectedIconColor = Palette.Red, selectedTextColor = Palette.Red,
                                indicatorColor = Palette.Yellow.copy(alpha = 0.45f), unselectedIconColor = Palette.Ink, unselectedTextColor = Palette.Ink))
                    }
                }
            }
            when (val s = engine.state) {
                EngineState.Loading -> Overlay {
                    CircularProgressIndicator(color = Palette.Ink)
                    Text("Loading 82 million weights…", style = display(18.sp))
                    Footnote("Everything runs on this device. Nothing leaves it.")
                }
                is EngineState.Failed -> Overlay {
                    Icon(Icons.Outlined.ErrorOutline, contentDescription = null, modifier = Modifier.size(40.dp))
                    Text("Couldn't load the model", style = display(18.sp))
                    Footnote(s.message, Modifier.padding(horizontal = 24.dp))
                }
                EngineState.Ready -> {}
            }
            if (about) {
                BackHandler { onAbout(false) }
                Box(Modifier.fillMaxSize().background(Palette.Paper).windowInsetsPadding(WindowInsets.systemBars)) {
                    AboutScreen(engine.config, engine.parameterCount, tips, actions) { onAbout(false) }
                }
            }
        }
    }
}

@Composable
private fun Overlay(content: @Composable ColumnScope.() -> Unit) {
    Column(Modifier.fillMaxSize().background(Palette.Paper), verticalArrangement = Arrangement.spacedBy(14.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally, content = content)
}
