#include "llama-tweak.h"

#include "common.h"
#include "log.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <sys/stat.h>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

static std::mutex g_mu;
static std::string g_active_model;
static json       g_cache_mem;
static std::string g_cache_path_loaded;
static bool       g_cache_loaded = false;
static std::string g_cache_path_override;

static std::string model_fingerprint(const std::string & path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) {
        return "missing";
    }
    return std::to_string((uint64_t) st.st_size) + ":" + std::to_string((uint64_t) st.st_mtime);
}

static std::string default_cache_path(const std::string & model_path) {
    fs::path stem = fs::path(model_path).stem();
    if (stem.empty()) {
        stem = "model";
    }
    return (fs::current_path() / ("llama-tweak-" + stem.string() + ".json")).string();
}

void llama_tweak_set_cache_path(const std::string & path) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_cache_path_override = path;
    g_cache_loaded        = false;
}

std::string llama_tweak_json_path_for_model(const std::string & model_path) {
    if (!g_cache_path_override.empty()) {
        return g_cache_path_override;
    }
    if (const char * env = std::getenv("LLAMA_TWEAK_CACHE")) {
        if (env[0] != '\0') {
            return env;
        }
    }
    return default_cache_path(model_path);
}

void llama_tweak_set_active_model(const std::string & model_path) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_active_model = model_path;
}

bool llama_tweak_runtime_enabled() {
    const char * v = std::getenv("LLAMA_TWEAK");
    if (v && std::atoi(v) != 0) {
        return true;
    }
    v = std::getenv("LLAMA_TWEAK_AUTO");
    return v && std::atoi(v) != 0;
}

static bool load_cache_locked(const std::string & model_path) {
    const std::string path = llama_tweak_json_path_for_model(model_path);
    if (g_cache_loaded && g_cache_path_loaded == path) {
        return true;
    }
    g_cache_loaded = false;
    g_cache_mem      = json::object();
    if (!fs::exists(path)) {
        return false;
    }
    try {
        std::ifstream in(path);
        g_cache_mem = json::parse(in);
    } catch (...) {
        return false;
    }
    const std::string fp = model_fingerprint(model_path);
    if (g_cache_mem.value("model_fingerprint", "") != fp) {
        return false;
    }
    g_cache_path_loaded = path;
    g_cache_loaded      = true;
    return true;
}

static int nearest_value(int req, const std::set<int> & values) {
    if (values.empty()) {
        return req;
    }
    int best   = *values.begin();
    int best_d = std::abs(best - req);
    for (int v : values) {
        const int d = std::abs(v - req);
        if (d < best_d) {
            best_d = d;
            best   = v;
        }
    }
    return best;
}

static bool entry_is_failed(const json & e) {
    return e.value("status", "") == "failed";
}

static bool entry_has_mean_tps(const json & e) {
    return e.contains("mean_tps") && e["mean_tps"].is_number();
}

static bool resolve_pp_tg_bucket(const json & entries, int pp, int tg, int & pp_use, int & tg_use) {
    std::set<int> pps;
    for (const auto & e : entries) {
        pps.insert(e.value("pp", -1));
    }
    pps.erase(-1);
    if (pps.empty()) {
        return false;
    }
    pp_use = nearest_value(pp, pps);

    std::set<int> tgs;
    for (const auto & e : entries) {
        if (e.value("pp", -1) == pp_use) {
            tgs.insert(e.value("tg", -1));
        }
    }
    tgs.erase(-1);
    if (tgs.empty()) {
        return false;
    }
    tg_use = nearest_value(tg, tgs);
    return true;
}

bool llama_tweak_resolve(const std::string & model_path, int pp, int tg, llama_tweak_plan & out) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!load_cache_locked(model_path)) {
        return false;
    }
    if (!g_cache_mem.contains("entries") || !g_cache_mem["entries"].is_array()) {
        return false;
    }

    const auto & entries = g_cache_mem["entries"];
    int          pp_use  = 0;
    int          tg_use  = 0;
    if (!resolve_pp_tg_bucket(entries, pp, tg, pp_use, tg_use)) {
        return false;
    }

    double best = -1.0;
    json   best_e;

    for (const auto & e : entries) {
        if (e.value("pp", -1) != pp_use || e.value("tg", -1) != tg_use) {
            continue;
        }
        if (entry_is_failed(e) || !entry_has_mean_tps(e)) {
            continue;
        }
        const double m = e.value("mean_tps", 0.0);
        if (m > best) {
            best   = m;
            best_e = e;
        }
    }

    if (best < 0) {
        return false;
    }

    out.resolved_pp    = pp_use;
    out.resolved_tg    = tg_use;
    out.selected_tag   = best_e.value("tag", "");
    out.expected_tps   = best;
    out.backend_kind   = best_e.value("backend_kind", "none");
    out.ggml_device    = best_e.value("ggml_device", "");
    out.openvino_device = best_e.value("openvino_device", "");
    out.openvino_stateful = best_e.value("openvino_stateful", 0);
    out.sycl_device_selector    = best_e.value("sycl_device_selector", "");
    return true;
}

bool llama_tweak_explain(const std::string & model_path, int pp, int tg) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!load_cache_locked(model_path)) {
        return false;
    }
    if (!g_cache_mem.contains("entries") || !g_cache_mem["entries"].is_array()) {
        return false;
    }

    const auto & entries = g_cache_mem["entries"];
    int          pp_use  = 0;
    int          tg_use  = 0;
    if (!resolve_pp_tg_bucket(entries, pp, tg, pp_use, tg_use)) {
        return false;
    }

    struct ranked_ok {
        double mean;
        json   e;
    };
    std::vector<ranked_ok> ok;
    std::vector<json>      failed;

    for (const auto & e : entries) {
        if (e.value("pp", -1) != pp_use || e.value("tg", -1) != tg_use) {
            continue;
        }
        if (entry_is_failed(e)) {
            failed.push_back(e);
        } else if (entry_has_mean_tps(e)) {
            ok.push_back({ e.value("mean_tps", 0.0), e });
        }
    }

    if (ok.empty() && failed.empty()) {
        return false;
    }

    std::sort(ok.begin(), ok.end(), [](const ranked_ok & a, const ranked_ok & b) {
        if (a.mean != b.mean) {
            return a.mean > b.mean;
        }
        return a.e.value("tag", "") < b.e.value("tag", "");
    });

    fprintf(stderr, "llama-tweak explain: request pp=%d tg=%d -> cache pp=%d tg=%d\n", pp, tg, pp_use, tg_use);

    int rank = 0;
    for (const auto & row : ok) {
        rank++;
        const json & e = row.e;
        const int    nruns = e.value("runs", 0);
        fprintf(stderr, "  %d. %s  %.2f tok/s", rank, e.value("tag", "?").c_str(), row.mean);
        fprintf(stderr, "  (runs=%d", nruns);
        if (e.contains("stddev_tps") && e["stddev_tps"].is_number() && nruns > 1) {
            fprintf(stderr, ", stddev=%.2f", e.value("stddev_tps", 0.0));
        }
        fprintf(stderr, ")\n");
    }

    if (!failed.empty()) {
        fprintf(stderr, "  failed (llama-bench did not complete successfully):\n");
        for (const auto & e : failed) {
            fprintf(stderr, "  - %s", e.value("tag", "?").c_str());
            if (e.contains("exit_code") && e["exit_code"].is_number()) {
                fprintf(stderr, "  exit=%d", e.value("exit_code", 0));
            }
            if (e.contains("attempted_runs") && e["attempted_runs"].is_number()) {
                fprintf(stderr, "  attempted_runs=%d", e.value("attempted_runs", 0));
            }
            fprintf(stderr, "\n");
        }
    }

    return true;
}

void llama_tweak_apply_env(const llama_tweak_plan & plan) {
    unsetenv("GGML_OPENVINO_DEVICE");
    unsetenv("GGML_OPENVINO_STATEFUL_EXECUTION");
    unsetenv("ONEAPI_DEVICE_SELECTOR");

    if (plan.backend_kind == "openvino") {
        if (!plan.openvino_device.empty()) {
            setenv("GGML_OPENVINO_DEVICE", plan.openvino_device.c_str(), 1);
        }
        setenv("GGML_OPENVINO_STATEFUL_EXECUTION", plan.openvino_stateful ? "1" : "0", 1);
    } else if (plan.backend_kind == "sycl") {
        if (!plan.sycl_device_selector.empty()) {
            setenv("ONEAPI_DEVICE_SELECTOR", plan.sycl_device_selector.c_str(), 1);
        }
    }
}

bool llama_tweak_save_cache_file(const std::string & model_path, const json & doc) {
    const std::string path = llama_tweak_json_path_for_model(model_path);
    try {
        const fs::path p(path);
        const fs::path parent = p.parent_path();
        if (!parent.empty()) {
            fs::create_directories(parent);
        }
        std::ofstream out(path);
        if (!out) {
            return false;
        }
        out << doc.dump(2) << "\n";
        if (!out.good()) {
            return false;
        }
    } catch (...) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    g_cache_mem         = doc;
    g_cache_path_loaded = path;
    g_cache_loaded      = true;
    return true;
}

json llama_tweak_load_or_empty(const std::string & model_path) {
    const std::string path = llama_tweak_json_path_for_model(model_path);
    if (fs::exists(path)) {
        try {
            std::ifstream in(path);
            json          doc = json::parse(in);
            doc["model_path"]        = model_path;
            doc["model_fingerprint"] = model_fingerprint(model_path);
            if (!doc.contains("entries")) {
                doc["entries"] = json::array();
            }
            return doc;
        } catch (...) {
        }
    }
    json doc;
    doc["schema_version"]    = 1;
    doc["model_path"]        = model_path;
    doc["model_fingerprint"] = model_fingerprint(model_path);
    doc["entries"]           = json::array();
    return doc;
}

void llama_tweak_merge_entry(json & doc, const json & entry) {
    auto & arr = doc["entries"];
    const std::string tag = entry.value("tag", "");
    json              kept = json::array();
    for (const auto & e : arr) {
        if (e.value("tag", "") != tag || e.value("pp", 0) != entry.value("pp", 0) ||
            e.value("tg", 0) != entry.value("tg", 0)) {
            kept.push_back(e);
        }
    }
    kept.push_back(entry);
    doc["entries"] = kept;
}

static int tweak_pp_from_env(const common_params & params) {
    if (const char * v = std::getenv("LLAMA_TWEAK_PP")) {
        return std::atoi(v);
    }
    if (const char * v = std::getenv("LLAMA_TWEAK_USE_N_BATCH")) {
        if (std::atoi(v) != 0 && params.n_batch > 0) {
            return params.n_batch;
        }
    }
    return 512;
}

static int tweak_tg_from_env() {
    if (const char * v = std::getenv("LLAMA_TWEAK_TG")) {
        return std::atoi(v);
    }
    return 128;
}

bool common_llama_tweak_prepare(common_params & params) {
    if (!params.llama_tweak_routing && !llama_tweak_runtime_enabled()) {
        return true;
    }
    if (params.model.path.empty()) {
        COM_ERR("%s\n", "llama-tweak needs a local model path (-m)");
        return false;
    }

    if (!params.llama_tweak_cache.empty()) {
        llama_tweak_set_cache_path(params.llama_tweak_cache);
    }

    const int pp = tweak_pp_from_env(params);
    const int tg = tweak_tg_from_env();

    llama_tweak_plan plan;
    if (!llama_tweak_resolve(params.model.path, pp, tg, plan)) {
        COM_ERR("llama-tweak: missing or stale cache for '%s' (pp=%d tg=%d)\n", params.model.path.c_str(), pp, tg);
        COM_ERR("Run: llama-tweak record -m %s --pp %d --tg %d\n", params.model.path.c_str(), pp, tg);
        return false;
    }

    llama_tweak_set_active_model(params.model.path);
    llama_tweak_apply_env(plan);

    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name(plan.ggml_device.c_str());
    if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        COM_ERR("llama-tweak: resolved device '%s' is not available\n", plan.ggml_device.c_str());
        return false;
    }

    params.devices.clear();
    params.devices.push_back(dev);
    params.devices.push_back(nullptr);

    if (plan.resolved_pp != pp || plan.resolved_tg != tg) {
        COM_INF("llama-tweak: request pp=%d tg=%d -> cache pp=%d tg=%d -> %s (%s, %.1f tok/s)\n", pp, tg,
                plan.resolved_pp, plan.resolved_tg, plan.selected_tag.c_str(), plan.ggml_device.c_str(),
                plan.expected_tps);
    } else {
        COM_INF("llama-tweak: pp=%d tg=%d -> %s (%s, %.1f tok/s expected)\n", pp, tg, plan.selected_tag.c_str(),
                plan.ggml_device.c_str(), plan.expected_tps);
    }
    return true;
}
