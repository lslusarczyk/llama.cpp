#include "llama-tweak.h"
#include "tweak-devices.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

static void usage() {
    fprintf(stderr,
            "usage: llama-tweak show\n"
            "       llama-tweak record (-m model.gguf | -hf user/model[:quant]) ...\n"
            "       llama-tweak explain (-m model.gguf | -hf user/model[:quant]) ...\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    if (strcmp(argv[1], "show") == 0) {
        ggml_backend_load_all();
        return llama_tweak_show_main(argc, argv);
    }
    return llama_tweak_record_main(argc, argv);
}
