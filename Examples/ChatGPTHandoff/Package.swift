// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ChatGPTHandoff",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "ChatGPTHandoff",
            dependencies: [
                .product(name: "AIKit", package: "aikitswift")
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        )
    ]
)
