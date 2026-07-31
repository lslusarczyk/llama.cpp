#pragma once

#include <string>
#include <vector>

struct llama_tweak_bench_config {
    std::string id;
    std::string backend_kind;
    std::string ggml_device;
    std::string openvino_device;
    int         openvino_stateful = 0;
    std::string display_line;
    std::string ov_cache_subdir;
};

std::vector<llama_tweak_bench_config> llama_tweak_enumerate_bench_configs();

std::vector<llama_tweak_bench_config> llama_tweak_filter_configs(
    const std::vector<llama_tweak_bench_config> & all,
    const std::string &                         backend_spec);

void llama_tweak_apply_bench_config_env(const llama_tweak_bench_config & cfg, int pp, int tg);

int llama_tweak_show_main(int argc, char ** argv);
