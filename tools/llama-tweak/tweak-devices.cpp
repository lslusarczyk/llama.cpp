#include "tweak-devices.h"

#include "ggml-backend.h"
#include "ggml-openvino.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <sstream>
#include <unordered_set>

static std::string strip_copy(const std::string & s) {
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char) s[a])) {
        a++;
    }
    size_t b = s.size();
    while (b > a && std::isspace((unsigned char) s[b - 1])) {
        b--;
    }
    return s.substr(a, b - a);
}

static std::string to_lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

static bool reg_available(const char * name) {
    return ggml_backend_reg_by_name(name) != nullptr;
}

static void append_ggml_devices(const char * reg_name, const char * id_prefix, std::vector<llama_tweak_bench_config> & out) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(reg_name);
    if (!reg) {
        return;
    }
    const size_t n = ggml_backend_reg_dev_count(reg);
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
        if (!dev) {
            continue;
        }
        llama_tweak_bench_config c;
        c.backend_kind  = to_lower(reg_name);
        c.id            = std::string(id_prefix) + std::to_string(i);
        c.ggml_device   = ggml_backend_dev_name(dev);
        c.display_line  = c.id + ": " + ggml_backend_dev_description(dev);
        c.ov_cache_subdir = c.id;
        out.push_back(std::move(c));
    }
}

static bool openvino_is_npu(const std::string & ov_dev) {
    return ov_dev == "NPU" || ov_dev.rfind("NPU", 0) == 0;
}

static void append_openvino_devices(std::vector<llama_tweak_bench_config> & out) {
    if (!reg_available(GGML_OPENVINO_NAME)) {
        return;
    }

    ggml_openvino_device_info infos[32];
    const int                 n = ggml_openvino_list_devices(infos, 32);
    if (n <= 0) {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name(GGML_OPENVINO_NAME);
        if (reg && ggml_backend_reg_dev_count(reg) > 0) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
            llama_tweak_bench_config c;
            c.backend_kind      = "openvino";
            c.id                = "openvino0";
            c.ggml_device       = dev ? ggml_backend_dev_name(dev) : "OPENVINO0";
            c.openvino_device   = "CPU";
            c.openvino_stateful = 0;
            c.display_line      = c.id + ": CPU (OpenVINO)";
            c.ov_cache_subdir   = c.id;
            out.push_back(c);
        }
        return;
    }

    const std::string ggml_ov = []() {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name(GGML_OPENVINO_NAME);
        if (reg && ggml_backend_reg_dev_count(reg) > 0) {
            return std::string(ggml_backend_dev_name(ggml_backend_reg_dev_get(reg, 0)));
        }
        return std::string("OPENVINO0");
    }();

    for (int i = 0; i < n; i++) {
        const std::string ov_name = infos[i].ov_device;
        const std::string desc    = infos[i].description;
        const bool        npu     = openvino_is_npu(ov_name);

        auto add_variant = [&](const std::string & suffix, int stateful) {
            llama_tweak_bench_config c;
            c.backend_kind      = "openvino";
            c.id                = "openvino" + std::to_string(i) + suffix;
            c.ggml_device       = ggml_ov;
            c.openvino_device   = ov_name;
            c.openvino_stateful = stateful;
            c.display_line      = c.id + ": " + ov_name + " - " + desc;
            c.ov_cache_subdir   = c.id;
            out.push_back(std::move(c));
        };

        if (npu) {
            add_variant("", 0);
        } else {
            add_variant("", 0);
            add_variant("_sf", 1);
        }
    }
}

std::vector<llama_tweak_bench_config> llama_tweak_enumerate_bench_configs() {
    std::vector<llama_tweak_bench_config> out;
    append_ggml_devices("Vulkan", "vulkan", out);
    append_openvino_devices(out);
    append_ggml_devices("SYCL", "sycl", out);
    return out;
}

std::vector<llama_tweak_bench_config> llama_tweak_filter_configs(
    const std::vector<llama_tweak_bench_config> & all,
    const std::string &                         backend_spec) {
    const std::string spec = to_lower(strip_copy(backend_spec));
    if (spec.empty() || spec == "all") {
        return all;
    }

    std::unordered_set<std::string> want;
    std::stringstream               ss(spec);
    std::string                     item;
    while (std::getline(ss, item, ',')) {
        item = to_lower(strip_copy(item));
        if (!item.empty()) {
            want.insert(item);
        }
    }

    std::vector<llama_tweak_bench_config> filtered;
    for (const auto & c : all) {
        if (want.count(c.id)) {
            filtered.push_back(c);
        }
    }
    return filtered;
}

static std::string shell_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

void llama_tweak_bench_config_env(const llama_tweak_bench_config & c, int pp, int tg, llama_tweak_bench_env & out) {
    out.unset_vars = {
        "GGML_OPENVINO_DEVICE",
        "GGML_OPENVINO_STATEFUL_EXECUTION",
        "ONEAPI_DEVICE_SELECTOR",
    };
    out.set_vars.clear();

    const std::string cache =
        "/tmp/llama_tweak_bench/" + c.ov_cache_subdir + "_" + std::to_string(pp) + "_" + std::to_string(tg);
    out.set_vars.emplace_back("GGML_OPENVINO_CACHE_DIR", cache);

    if (c.backend_kind == "openvino") {
        if (!c.openvino_device.empty()) {
            out.set_vars.emplace_back("GGML_OPENVINO_DEVICE", c.openvino_device);
        }
        out.set_vars.emplace_back("GGML_OPENVINO_STATEFUL_EXECUTION", c.openvino_stateful ? "1" : "0");
    }
}

void llama_tweak_apply_bench_config_env(const llama_tweak_bench_config & c, int pp, int tg) {
    llama_tweak_bench_env env;
    llama_tweak_bench_config_env(c, pp, tg, env);
    for (const auto & u : env.unset_vars) {
        unsetenv(u.c_str());
    }
    for (const auto & kv : env.set_vars) {
        setenv(kv.first.c_str(), kv.second.c_str(), 1);
    }
}

std::string llama_tweak_format_bench_shell_command(
    const std::string &              llama_bench_path,
    const llama_tweak_bench_env &    env,
    const std::vector<std::string> & llama_bench_args) {
    std::ostringstream cmd;
    for (const auto & u : env.unset_vars) {
        cmd << "env -u " << u << " ";
    }
    for (const auto & kv : env.set_vars) {
        cmd << kv.first << "=" << shell_quote(kv.second) << " ";
    }
    cmd << shell_quote(llama_bench_path);
    for (const auto & a : llama_bench_args) {
        cmd << " " << shell_quote(a);
    }
    return cmd.str();
}
