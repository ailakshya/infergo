{
  "targets": [
    {
      "target_name": "infergo",
      "sources": ["src/infergo.c"],
      "include_dirs": [
        "../../cpp/include",
        "<!(node -p \"require('node-addon-api').include_dir\")"
      ],
      "libraries": [
        "-linfer_api",
        "-L<(module_root_dir)/../../build/cpp/api",
        "-Wl,-rpath,<(module_root_dir)/../../build/cpp/api"
      ],
      "cflags": ["-std=c11", "-Wall", "-Wextra"],
      "conditions": [
        ["OS=='linux'", {
          "libraries": [
            "-Wl,-rpath,'$$ORIGIN/../../build/cpp/api'"
          ]
        }],
        ["OS=='mac'", {
          "xcode_settings": {
            "OTHER_CFLAGS": ["-std=c11"],
            "OTHER_LDFLAGS": [
              "-Wl,-rpath,@loader_path/../../build/cpp/api"
            ]
          }
        }]
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"]
    }
  ]
}
