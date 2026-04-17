// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "Infergo",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "Infergo", targets: ["Infergo"]),
    ],
    targets: [
        // C shim that wraps infer_api.h
        .target(
            name: "CInfergo",
            path: "Sources/CInfergo",
            cSettings: [
                // Point at the infergo C header in the build tree.
                // Override with: swift build -Xcc -I/path/to/cpp/include
                .unsafeFlags(["-I../../cpp/include"]),
            ],
            linkerSettings: [
                .linkedLibrary("infer_api"),
            ]
        ),
        // Swift wrapper library
        .target(
            name: "Infergo",
            dependencies: ["CInfergo"],
            path: "Sources/Infergo"
        ),
    ]
)
