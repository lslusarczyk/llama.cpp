#pragma once

#include <string>
#include <vector>

struct llama_tweak_bench_config {
    std::string id;
    std::string backend_kind;
    std::string ggml_device;
    std::string openvino_device;
    int         openvino_stateful = 0;
    int         sycl_enable_graph = 0;
    bool        sycl_native_graph = false;
    std::string display_line;
    std::string ov_cache_subdir;
};

std::vector<llama_tweak_bench_config> llama_tweak_enumerate_bench_configs();

std::vector<llama_tweak_bench_config> llama_tweak_filter_configs(
    const std::vector<llama_tweak_bench_config> & all,
    const std::string &                         backend_spec);

struct llama_tweak_bench_env {
    std::vector<std::string>              unset_vars;
    std::vector<std::pair<std::string, std::string>> set_vars;
};

void llama_tweak_bench_config_env(const llama_tweak_bench_config & cfg, int pp, int tg, llama_tweak_bench_env & out);

void llama_tweak_apply_bench_config_env(const llama_tweak_bench_config & cfg, int pp, int tg);

std::string llama_tweak_format_bench_shell_command(
    const std::string &               llama_bench_path,
    const llama_tweak_bench_env &     env,
    const std::vector<std::string> &  llama_bench_args);

int llama_tweak_show_main(int argc, char ** argv);
