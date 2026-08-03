#include "llama-tweak.h"
#include "tweak-devices.h"

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits.h>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using json     = nlohmann::ordered_json;

static constexpr size_t k_bench_log_max = 256 * 1024;

static std::string resolve_llama_bench_path() {
    if (const char * env = std::getenv("LLAMA_BENCH")) {
        if (env[0] != '\0' && access(env, X_OK) == 0) {
            return env;
        }
    }
    char self[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n > 0) {
        self[n] = '\0';
        const fs::path candidate = fs::path(self).parent_path() / "llama-bench";
        if (access(candidate.c_str(), X_OK) == 0) {
            return candidate.string();
        }
    }
    return "llama-bench";
}

static std::vector<std::string> bench_argv(const std::string & model, int pp, int tg, const std::string & dev) {
    return {
        "-m", model, "-r", "1", "--no-warmup", "-p", "0", "-n", "0",
        "-pg", std::to_string(pp) + "," + std::to_string(tg), "-o", "jsonl", "--device", dev, "-ngl", "999"};
}

struct bench_run_result {
    bool        ok = false;
    double      tps = 0.0;
    int         exit_code = -1;
    std::string log;
    std::string replay_cmd;
};

static bool parse_bench_jsonl(const std::string & output, const std::string & model, int pp, int tg, double & out_tps) {
    const fs::path model_base = fs::path(model).filename();

    std::istringstream stream(output);
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

static bench_run_result run_bench_subprocess(
    const std::string &              llama_bench_path,
    const std::string &              model,
    int                              pp,
    int                              tg,
    const llama_tweak_bench_config & c) {
    bench_run_result result;

    llama_tweak_bench_env env;
    llama_tweak_bench_config_env(c, pp, tg, env);

    const auto args_s = bench_argv(model, pp, tg, c.ggml_device);
    result.replay_cmd = llama_tweak_format_bench_shell_command(llama_bench_path, env, args_s);

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        result.log = "llama-tweak: pipe() failed\n";
        return result;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        result.log = "llama-tweak: fork() failed\n";
        return result;
    }

    if (pid == 0) {
        close(pipefd[0]);
        for (const auto & u : env.unset_vars) {
            unsetenv(u.c_str());
        }
        for (const auto & kv : env.set_vars) {
            setenv(kv.first.c_str(), kv.second.c_str(), 1);
        }
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        std::vector<std::string> storage;
        storage.push_back(llama_bench_path);
        for (const auto & a : args_s) {
            storage.push_back(a);
        }
        std::vector<char *> argv;
        for (auto & s : storage) {
            argv.push_back(s.data());
        }
        argv.push_back(nullptr);

        execv(llama_bench_path.c_str(), argv.data());
        _exit(127);
    }

    close(pipefd[1]);
    result.log.reserve(4096);
    char chunk[4096];
    while (result.log.size() < k_bench_log_max) {
        const ssize_t n = read(pipefd[0], chunk, sizeof(chunk));
        if (n <= 0) {
            break;
        }
        result.log.append(chunk, (size_t) n);
    }
    close(pipefd[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        result.log += "\nllama-tweak: waitpid() failed\n";
        return result;
    }

    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
        result.log += "\nllama-tweak: llama-bench terminated by signal " + std::to_string(WTERMSIG(status)) + "\n";
    }

    if (result.log.size() >= k_bench_log_max) {
        result.log += "\n... (log truncated)\n";
    }

    if (result.exit_code != 0) {
        return result;
    }

    result.ok = parse_bench_jsonl(result.log, model, pp, tg, result.tps);
    return result;
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

static json bench_entry_base(const llama_tweak_bench_config & c, int pp, int tg) {
    json e;
    e["tag"]                     = c.id;
    e["pp"]                      = pp;
    e["tg"]                      = tg;
    e["backend_kind"]            = c.backend_kind;
    e["ggml_device"]             = c.ggml_device;
    e["openvino_device"]         = c.openvino_device;
    e["openvino_stateful"]       = c.openvino_stateful;
    e["sycl_device_selector"]    = "";
    return e;
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

static void print_backend_failure(const std::string & config_id, int pp, int tg, const bench_run_result & run) {
    fprintf(stderr, "llama-tweak: backend %s failed for pp=%d tg=%d", config_id.c_str(), pp, tg);
    if (run.exit_code >= 0) {
        fprintf(stderr, " (exit %d)", run.exit_code);
    }
    fprintf(stderr, "; skipping this run\n");
    if (!run.log.empty()) {
        fprintf(stderr, "llama-bench log:\n%s", run.log.c_str());
        if (run.log.back() != '\n') {
            fprintf(stderr, "\n");
        }
    }
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
        if (!llama_tweak_explain(model, pp, tg)) {
            fprintf(stderr, "llama-tweak: no cache for %s (pp=%d tg=%d). Run: llama-tweak record -m ...\n", model.c_str(),
                    pp, tg);
            return 1;
        }
        return 0;
    }

    if (cmd != "record") {
        usage();
        return 1;
    }

    const std::string llama_bench_path = resolve_llama_bench_path();
    if (llama_bench_path != "llama-bench" && access(llama_bench_path.c_str(), X_OK) != 0) {
        fprintf(stderr, "llama-tweak: llama-bench not found at %s (set LLAMA_BENCH or build llama-bench next to llama-tweak)\n",
                llama_bench_path.c_str());
        return 1;
    }
    fprintf(stderr, "llama-tweak: using llama-bench at %s\n", llama_bench_path.c_str());

    std::vector<int> pps;
    parse_pg_list(pp_list.c_str(), pps);
    const int tg = std::max(0, std::atoi(tg_val.c_str()));
    if (pps.empty()) {
        pps.push_back(512);
    }

    json doc = llama_tweak_load_or_empty(model);

    ggml_backend_load_all();
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
                fprintf(stderr, "=== NPU probe %s ===\n", c.id.c_str());
                const bench_run_result probe = run_bench_subprocess(llama_bench_path, model, 8, 0, c);
                fprintf(stderr, "replay:\n  %s\n", probe.replay_cmd.c_str());
                if (!probe.ok) {
                    fprintf(stderr, "skip %s (NPU probe failed)\n", c.id.c_str());
                    print_backend_failure(c.id, 8, 0, probe);
                    continue;
                }
            }
            std::vector<double> samples;
            fprintf(stderr, "=== %s pp=%d tg=%d (%d runs) ===\n", c.id.c_str(), pp, tg, runs);
            bench_run_result last_fail;
            bool             have_fail = false;
            for (int r = 0; r < runs; ++r) {
                const bench_run_result run = run_bench_subprocess(llama_bench_path, model, pp, tg, c);
                fprintf(stderr, "replay:\n  %s\n", run.replay_cmd.c_str());
                if (!run.ok) {
                    fprintf(stderr, "  run %d: FAIL\n", r + 1);
                    last_fail = run;
                    have_fail = true;
                    continue;
                }
                samples.push_back(run.tps);
                fprintf(stderr, "  run %d: %.2f tok/s\n", r + 1, run.tps);
            }
            if (samples.empty()) {
                fprintf(stderr, "llama-tweak: no successful runs for %s pp=%d tg=%d (backend unavailable)\n", c.id.c_str(),
                        pp, tg);
                if (have_fail) {
                    print_backend_failure(c.id, pp, tg, last_fail);
                }
                if (have_fail && !last_fail.replay_cmd.empty()) {
                    fprintf(stderr, "last replay:\n  %s\n", last_fail.replay_cmd.c_str());
                }
                json e = bench_entry_base(c, pp, tg);
                e["status"]         = "failed";
                e["attempted_runs"] = runs;
                if (have_fail && last_fail.exit_code >= 0) {
                    e["exit_code"] = last_fail.exit_code;
                }
                llama_tweak_merge_entry(doc, e);
                continue;
            }
            json e = bench_entry_base(c, pp, tg);
            e["status"]   = "ok";
            e["mean_tps"] = mean_vec(samples);
            e["runs"]     = (int) samples.size();
            if (samples.size() >= 2) {
                e["stddev_tps"] = stdev_vec(samples);
            }
            if (have_fail) {
                e["bench_failed_runs"] = runs - (int) samples.size();
            }
            llama_tweak_merge_entry(doc, e);
        }
    }

    if (!llama_tweak_save_cache_file(model, doc)) {
        const std::string path = llama_tweak_json_path_for_model(model);
        fprintf(stderr,
                "failed to write %s (check that the directory exists and is writable; use --output with a full path if needed)\n",
                path.c_str());
        return 1;
    }
    fprintf(stderr, "wrote %s\n", llama_tweak_json_path_for_model(model).c_str());
    return 0;
}
