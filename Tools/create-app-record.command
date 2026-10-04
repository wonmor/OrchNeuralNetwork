#!/bin/zsh
# Creates the App Store Connect record for Orch Neural Network via fastlane produce.
# Prompts for the Apple ID password and 2FA code; nothing is stored outside the macOS Keychain.
export PATH="/opt/homebrew/bin:$PATH"
fastlane produce create \
  --username wonmor@yahoo.com \
  --app_identifier com.johnseong.OrchNeuralNetwork \
  --app_name "Orch Neural Network" \
  --language en-US \
  --sku orchneuralnetwork \
  --team_id Z64KRUX3W3
echo
echo "Done. You can close this window and tell Claude the record was created."
