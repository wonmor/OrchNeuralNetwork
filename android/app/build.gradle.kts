import java.io.FileInputStream
import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("app.cash.paparazzi")
}

// Release signing: local.properties (never committed) names a properties file with
// storeFile / storePassword / keyAlias / keyPassword. Without it, release builds use the debug key.
val localProps = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) FileInputStream(f).use { load(it) }
}
val signingFile = localProps.getProperty("signing.properties")?.let { File(it) }?.takeIf { it.exists() }
val signingProps = Properties().apply { signingFile?.let { f -> FileInputStream(f).use { load(it) } } }

android {
    namespace = "com.johnseong.orchneuralnetwork"
    compileSdk = 36
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.johnseong.orchneuralnetwork"
        minSdk = 26
        targetSdk = 36
        versionCode = 2
        versionName = "1.0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // 64-bit only: the 164 MB model plus activations needs a 64-bit address space anyway.
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }

    // The weights, vocabulary and oracle are the same files the iOS app bundles (repo-root Model/).
    sourceSets["main"].assets.srcDirs("../../Model")
    // The weight file must stay uncompressed in the APK so it can be memory-mapped.
    androidResources { noCompress += listOf("tsw") }

    signingConfigs {
        if (signingFile != null) {
            create("release") {
                val store = signingProps.getProperty("storeFile")
                storeFile = File(store).takeIf { it.isAbsolute } ?: File(signingFile.parentFile, store)
                storePassword = signingProps.getProperty("storePassword")
                keyAlias = signingProps.getProperty("keyAlias")
                keyPassword = signingProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = if (signingFile != null) signingConfigs.getByName("release") else signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { compose = true; buildConfig = true }
    testOptions { unitTests.isReturnDefaultValues = true }
    packaging { jniLibs { useLegacyPackaging = false } }
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.9.3")

    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    debugImplementation("androidx.compose.ui:ui-tooling")
    implementation("androidx.compose.ui:ui-tooling-preview")

    implementation("com.android.billingclient:billing-ktx:8.0.0")

    testImplementation("junit:junit:4.13.2")
    // org.json is stubbed in android.jar; the JVM parity tests need the real thing.
    testImplementation("org.json:json:20240303")

    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
