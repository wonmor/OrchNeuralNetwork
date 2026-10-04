fastlane documentation
----

# Installation

Make sure you have the latest version of the Xcode command line tools installed:

```sh
xcode-select --install
```

For _fastlane_ installation instructions, see [Installing _fastlane_](https://docs.fastlane.tools/#installing-fastlane)

# Available Actions

## iOS

### ios age_rating

```sh
[bundle exec] fastlane ios age_rating
```

Set the age rating questionnaire to all-None (4+), including the 2025 questions fastlane's deliver does not know

### ios privacy

```sh
[bundle exec] fastlane ios privacy
```

Declare 'Data Not Collected' in the App Privacy section

### ios status

```sh
[bundle exec] fastlane ios status
```

Show build processing and version state

### ios submit

```sh
[bundle exec] fastlane ios submit
```

Submit version 1.0 for review using the latest processed build

### ios debug

```sh
[bundle exec] fastlane ios debug
```

Dump what App Store Connect actually has for this app

### ios pricing

```sh
[bundle exec] fastlane ios pricing
```

Set price to Free (USA base territory) and make the app available in all territories

----

This README.md is auto-generated and will be re-generated every time [_fastlane_](https://fastlane.tools) is run.

More information about _fastlane_ can be found on [fastlane.tools](https://fastlane.tools).

The documentation of _fastlane_ can be found on [docs.fastlane.tools](https://docs.fastlane.tools).
