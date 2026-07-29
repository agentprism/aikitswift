// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Manifold",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "Manifold", targets: ["Manifold"])
    ],
    targets: [
        .target(
            name: "Manifold",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ManifoldTests",
            dependencies: ["Manifold"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
