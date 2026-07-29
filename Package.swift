// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AIKitSwift",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "AIKit", targets: ["AIKit"])
    ],
    targets: [
        .target(
            name: "AIKit",
            resources: [.copy("Catalog")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "AIKitTests",
            dependencies: ["AIKit"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
