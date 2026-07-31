#include "llama-tweak.h"
#include "tweak-devices.h"

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-bench-api.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using json     = nlohmann::ordered_json;

static bool run_bench_capture(const std::string & model, int pp, int tg, const llama_tweak_bench_config & c,
                              double & out_tps) {
    llama_tweak_apply_bench_config_env(c, pp, tg);

    const std::string dev = c.ggml_device;

    std::vector<std::string> args_s = {
        "llama-bench", "-m", model, "-r", "1", "--no-warmup", "-p", "0", "-n", "0",
        "-pg", std::to_string(pp) + "," + std::to_string(tg), "-o", "jsonl", "--device", dev, "-ngl", "999"};
    std::vector<char *> argv;
    for (auto & s : args_s) {
        argv.push_back(s.data());
    }
    argv.push_back(nullptr);

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return false;
    }

    FILE * orig_out = stdout;
    fflush(stdout);
    stdout = fdopen(pipefd[1], "w");
    if (!stdout) {
        close(pipefd[0]);
        close(pipefd[1]);
        stdout = orig_out;
        return false;
    }

    const int rc = llama_bench((int) argv.size() - 1, argv.data());
    fflush(stdout);
    fclose(stdout);
    stdout = orig_out;

    std::string line;
    {
        char    chunk[4096];
        ssize_t n;
        while ((n = read(pipefd[0], chunk, sizeof(chunk))) > 0) {
            line.append(chunk, (size_t) n);
        }
    }
    close(pipefd[0]);

    if (rc != 0) {
        return false;
    }

    const fs::path model_base = fs::path(model).filename();

    std::istringstream stream(line);
    std::string        one;
    while (std::getline(stream, one)) {
        const auto pos = one.find('{');
        if (pos == std::string::npos) {
            continue;
        }
        try {
            json j = json::parse(one.substr(pos));
            if (!j.contains("build_commit") || !j.contains("avg_ts")) {
                continue;
            }
            if (j.value("n_prompt", -1) != pp || j.value("n_gen", -1) != tg) {
                continue;
            }
            const std::string mf = j.value("model_filename", "");
            if (!mf.empty() && fs::path(mf).filename() != model_base) {
                continue;
            }
            out_tps = j.value("avg_ts", 0.0);
            if (out_tps > 0.0 && out_tps < 100000.0) {
                return true;
            }
        } catch (...) {
        }
    }
    return false;
}

static void parse_pg_list(const char * s, std::vector<int> & out) {
    out.clear();
    std::stringstream ss(s);
    std::string       item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::atoi(item.c_str()));
        }
    }
}

static double mean_vec(const std::vector<double> & v) {
    if (v.empty()) {
        return 0.0;
    }
    double s = 0;
    for (double x : v) {
        s += x;
    }
    return s / (double) v.size();
}

static double stdev_vec(const std::vector<double> & v) {
    if (v.size() < 2) {
        return 0.0;
    }
    const double m = mean_vec(v);
    double       s = 0;
    for (double x : v) {
        const double d = x - m;
        s += d * d;
    }
    return std::sqrt(s / (double) (v.size() - 1));
}

static void usage() {
    fprintf(stderr,
            "usage: llama-tweak show\n"
            "       llama-tweak record (-m model.gguf | -hf user/model[:quant]) [--backend all|id,...]\n"
            "           [--pp 128,512] [--tg 128] [--runs 3] [--output path.json]\n"
            "       llama-tweak explain (-m model.gguf | -hf user/model[:quant]) [--pp N] [--tg N]\n");
}

static bool resolve_tweak_model(common_params & params, std::string & model_path, std::string & err) {
    if (const char * tok = std::getenv("HF_TOKEN")) {
        if (params.hf_token.empty() && tok[0] != '\0') {
            params.hf_token = tok;
        }
    }
    if (!params.model.hf_repo.empty()) {
        if (params.offline) {
            fprintf(stderr, "llama-tweak: using Hugging Face cache for %s (--offline)\n",
                    params.model.hf_repo.c_str());
        } else {
            fprintf(stderr,
                    "llama-tweak: resolving %s (HF API, then download if missing; "
                    "file downloads show a progress bar on a TTY)...\n",
                    params.model.hf_repo.c_str());
        }
    }
    common_init();
    if (!common_params_resolve_model(params, LLAMA_EXAMPLE_BENCH, &err)) {
        return false;
    }
    model_path = params.model.path;
    return true;
}

int llama_tweak_record_main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string cmd = argv[1];
    common_params cparams;
    std::string model;
    std::string pp_list = "512";
    std::string tg_val  = "128";
    std::string out_path;
    std::string backend_spec = "all";
    int         runs         = 3;

    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            cparams.model.path = argv[++i];
        } else if ((!strcmp(argv[i], "-hf") || !strcmp(argv[i], "-hfr") || !strcmp(argv[i], "--hf-repo")) && i + 1 < argc) {
            cparams.model.hf_repo = argv[++i];
        } else if ((!strcmp(argv[i], "-hff") || !strcmp(argv[i], "--hf-file")) && i + 1 < argc) {
            cparams.model.hf_file = argv[++i];
        } else if ((!strcmp(argv[i], "-hft") || !strcmp(argv[i], "--hf-token")) && i + 1 < argc) {
            cparams.hf_token = argv[++i];
        } else if (!strcmp(argv[i], "--offline")) {
            cparams.offline = true;
        } else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            backend_spec = argv[++i];
        } else if (!strcmp(argv[i], "--pp") && i + 1 < argc) {
            pp_list = argv[++i];
        } else if (!strcmp(argv[i], "--tg") && i + 1 < argc) {
            tg_val = argv[++i];
        } else if ((!strcmp(argv[i], "--output") || !strcmp(argv[i], "-o")) && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "--runs") && i + 1 < argc) {
            runs = std::max(1, std::atoi(argv[++i]));
        }
    }

    {
        std::string err;
        if (!resolve_tweak_model(cparams, model, err)) {
            fprintf(stderr, "llama-tweak: %s\n", err.c_str());
            usage();
            return 1;
        }
    }

    if (!out_path.empty()) {
        llama_tweak_set_cache_path(out_path);
    } else if (const char * env = std::getenv("LLAMA_TWEAK_CACHE")) {
        if (env[0] != '\0') {
            llama_tweak_set_cache_path(env);
        }
    }

    if (cmd == "explain") {
        int pp = std::getenv("LLAMA_TWEAK_PP") ? std::atoi(std::getenv("LLAMA_TWEAK_PP")) : 512;
        int tg = std::getenv("LLAMA_TWEAK_TG") ? std::atoi(std::getenv("LLAMA_TWEAK_TG")) : 128;
        for (int j = 2; j < argc; ++j) {
            if (!strcmp(argv[j], "--pp") && j + 1 < argc) {
                pp = std::atoi(argv[++j]);
            } else if (!strcmp(argv[j], "--tg") && j + 1 < argc) {
                tg = std::atoi(argv[++j]);
            }
        }
        llama_tweak_plan plan;
        if (!llama_tweak_resolve(model, pp, tg, plan)) {
            fprintf(stderr, "llama-tweak: no cache for %s (pp=%d tg=%d). Run: llama-tweak record -m ...\n", model.c_str(),
                    pp, tg);
            return 1;
        }
        fprintf(stderr, "best: %s backend=%s ggml_dev=%s cache_pp=%d cache_tg=%d expected=%.2f tok/s\n",
                plan.selected_tag.c_str(), plan.backend_kind.c_str(), plan.ggml_device.c_str(), plan.resolved_pp,
                plan.resolved_tg, plan.expected_tps);
        return 0;
    }

    if (cmd != "record") {
        usage();
        return 1;
    }

    std::vector<int> pps;
    parse_pg_list(pp_list.c_str(), pps);
    const int tg = std::max(0, std::atoi(tg_val.c_str()));
    if (pps.empty()) {
        pps.push_back(512);
    }

    json doc = llama_tweak_load_or_empty(model);

    const auto all_cfg = llama_tweak_enumerate_bench_configs();
    const auto cases   = llama_tweak_filter_configs(all_cfg, backend_spec);
    if (cases.empty()) {
        fprintf(stderr, "llama-tweak: no matching --backend configs (run: llama-tweak show)\n");
        return 1;
    }

    for (int pp : pps) {
        for (const auto & c : cases) {
            if (c.backend_kind == "openvino" &&
                (c.openvino_device == "NPU" || c.openvino_device.rfind("NPU", 0) == 0)) {
                double probe = 0;
                if (!run_bench_capture(model, 8, 0, c, probe)) {
                    fprintf(stderr, "skip %s (NPU probe failed)\n", c.id.c_str());
                    continue;
                }
            }
            std::vector<double> samples;
            fprintf(stderr, "=== %s pp=%d tg=%d (%d runs) ===\n", c.id.c_str(), pp, tg, runs);
            for (int r = 0; r < runs; ++r) {
                double tps = 0;
                if (!run_bench_capture(model, pp, tg, c, tps)) {
                    fprintf(stderr, "  run %d: FAIL\n", r + 1);
                    continue;
                }
                samples.push_back(tps);
                fprintf(stderr, "  run %d: %.2f tok/s\n", r + 1, tps);
            }
            if (samples.empty()) {
                continue;
            }
            json e;
            e["tag"]                      = c.id;
            e["pp"]                       = pp;
            e["tg"]                       = tg;
            e["backend_kind"]             = c.backend_kind;
            e["ggml_device"]              = c.ggml_device;
            e["openvino_device"]          = c.openvino_device;
            e["openvino_stateful"]        = c.openvino_stateful;
            e["openvino_phase_split"]     = false;
            e["openvino_prefill_device"]  = "";
            e["openvino_decode_device"]   = "";
            e["sycl_device_selector"]     = "";
            e["mean_tps"]                 = mean_vec(samples);
            e["stddev_tps"]               = stdev_vec(samples);
            e["runs"]                     = (int) samples.size();
            llama_tweak_merge_entry(doc, e);
        }
    }

    if (!llama_tweak_save_cache_file(model, doc)) {
        fprintf(stderr, "failed to write %s\n", llama_tweak_json_path_for_model(model).c_str());
        return 1;
    }
    fprintf(stderr, "wrote %s\n", llama_tweak_json_path_for_model(model).c_str());
    return 0;
}
