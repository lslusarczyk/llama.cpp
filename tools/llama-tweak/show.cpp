#include "tweak-devices.h"

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

int llama_tweak_show_main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    const auto configs = llama_tweak_enumerate_bench_configs();
    if (configs.empty()) {
        fprintf(stderr, "llama-tweak: no Intel backends (Vulkan / OpenVINO / SYCL) in this build.\n");
        return 1;
    }

    std::unordered_map<std::string, std::vector<const llama_tweak_bench_config *>> by_backend;
    for (const auto & c : configs) {
        by_backend[c.backend_kind].push_back(&c);
    }

    const char * order[] = { "vulkan", "openvino", "sycl" };
    for (const char * bk : order) {
        auto it = by_backend.find(bk);
        if (it == by_backend.end()) {
            continue;
        }
        printf("[%s]\n", bk);
        for (const auto * p : it->second) {
            printf("  %s\n", p->display_line.c_str());
        }
        printf("\n");
    }

    printf("configurations (record --backend all):\n");
    for (const char * bk : order) {
        auto it = by_backend.find(bk);
        if (it == by_backend.end() || it->second.empty()) {
            continue;
        }
        for (size_t i = 0; i < it->second.size(); i++) {
            if (i > 0) {
                printf(", ");
            }
            printf("%s", it->second[i]->id.c_str());
        }
        printf("\n");
    }

    return 0;
}
