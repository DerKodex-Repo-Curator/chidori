#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <slurm/spank.h>

SPANK_PLUGIN(juice, 1)

static char juice_cmd[256] = "";
static char juice_data[256] = "";

static char juice_pool[64] = "";
static int juice_num_gpus = 1;
static int juice_vram = 0;

int juice_init(int argc, char **argv);
int juice_setenv(spank_t sp, const char* name, const char * value);
void generate_random_string(char *str, size_t length);

int handle_option_juice_pool(int val, const char *value, int remote);
static struct spank_option juice_pool_option = {
    .name = "juice-pool",
    .arginfo = "<pool name|pool id>",
    .usage = "GPU pool to use",
    .has_arg = 1,
    .val = 0,
    .cb = handle_option_juice_pool
};

int handle_option_juice_num_gpus(int val, const char *value, int remote);
static struct spank_option juice_num_gpus_option = {
    .name = "juice-num-gpus",
    .arginfo = "<gpu-count>",
    .usage = "The number of GPUs to request",
    .has_arg = 1,
    .val = 0,
    .cb = handle_option_juice_num_gpus
};

int handle_option_juice_vram(int val, const char *value, int remote);
static struct spank_option juice_vram_option = {
    .name = "juice-vram",
    .arginfo = "<vram>",
    .usage = "VRAM in GiB",
    .has_arg = 1,
    .val = 0,
    .cb = handle_option_juice_vram
};

int slurm_spank_init(spank_t sp, int argc, char **argv) {
    int result = 0;
    result = juice_init(argc, argv);
    if (result != ESPANK_SUCCESS) {
        return result;
    }

    result = spank_option_register(sp, &juice_pool_option);
    if (result != ESPANK_SUCCESS) {
        return result;
    }

    result = spank_option_register(sp, &juice_num_gpus_option);
    if (result != ESPANK_SUCCESS) {
        return result;
    }

    result = spank_option_register(sp, &juice_vram_option);
    if (result != ESPANK_SUCCESS) {
        return result;
    }

    return ESPANK_SUCCESS;
}

int slurm_spank_init_post_opt(spank_t sp, __attribute__((unused)) int argc, __attribute__((unused)) char** argv) {
    char encryption_key[33];
    generate_random_string(encryption_key, sizeof(encryption_key) - 1);
    if (juice_setenv(sp, "JUICE_SESSION_ENCRYPTION_KEY", encryption_key) != ESPANK_SUCCESS) {
        slurm_error("Failed to set Juice encryption key");
        return ESPANK_ERROR;
    }

    return ESPANK_SUCCESS;
}

int slurm_spank_task_init(spank_t sp, __attribute__((unused)) int argc, __attribute__((unused)) char **argv) {
    if (strlen(juice_pool) == 0) {
        // Not using juice, nothing to do
        return ESPANK_SUCCESS;
    }

    unsigned int job_id = 0;
    if (spank_get_item(sp, S_JOB_ID, &job_id) != ESPANK_SUCCESS) {
        slurm_error("Failed to get SLURM job id");
        return ESPANK_ERROR;
    }

    char session_path[1024];
    snprintf(session_path, sizeof(session_path) - 1, "%s/sessions/job-%d.cfg", juice_data, job_id);

    const char *prepend_args[] = {
        juice_cmd,
        "--no-banner",
        "run",
        "--import-session",
        session_path
    };
    spank_prepend_task_argv(sp, sizeof(prepend_args) / sizeof(prepend_args[0]), prepend_args);
    return 0;
}

int handle_option_juice_pool(__attribute__((unused)) int val, const char *value, __attribute__((unused)) int remote) {
    if (value) {
        strncpy(juice_pool, value, sizeof(juice_pool) - 1);
        juice_pool[sizeof(juice_pool) - 1] = '\0';
    }
    return ESPANK_SUCCESS;
}

int handle_option_juice_num_gpus(__attribute__((unused)) int val, const char *value, __attribute__((unused)) int remote) {
    if (value) {
        juice_num_gpus = atoi(value);
        if (juice_num_gpus == 0) {
            slurm_error("Invalid gpu count %s", value);
            return ESPANK_ERROR;
        }
    }
    return ESPANK_SUCCESS;
}

int handle_option_juice_vram(__attribute__((unused)) int val, const char *value, __attribute__((unused)) int remote) {
    if (value) {
        juice_vram = atoi(value);
        if (juice_vram < 1) {
            slurm_error("Invalid vram size %s", value);
            return ESPANK_ERROR;
        }
    }
    return ESPANK_SUCCESS;
}

int juice_init(int argc, char **argv) {
    if (argc != 2) {
        slurm_error("Juice Plugin requires two arguments: \"required juice.so /path/to/juice /path/to/shared/data/folder\"");
        return ESPANK_ERROR;
    }

    struct stat fileStat;
    if (strcmp(&argv[0][strlen(argv[0])-5], "juice") != 0 || stat(argv[0], &fileStat) < 0) {
        slurm_error("Invalid juice path %s", argv[0]);
        return ESPANK_ERROR;
    }
    strncpy(juice_cmd, argv[0], sizeof(juice_cmd)-1);

    if (stat(argv[1], &fileStat) < 0) {
        slurm_error("Invalid juice data path %s", argv[1]);
        return ESPANK_ERROR;
    }
    strncpy(juice_data, argv[1], sizeof(juice_data)-1);
    return ESPANK_SUCCESS;
}

int juice_setenv(spank_t sp, const char* name, const char* value) {
    int result = 0;
    result = spank_job_control_setenv(sp, name, value, 1);
    if (result != ESPANK_SUCCESS) {
        return result;
    }
    result = setenv(name, value, 1);
    if (result != ESPANK_SUCCESS) {
        return result;
    }

    return ESPANK_SUCCESS;
}

void generate_random_string(char *str, size_t length) {
    const char charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    size_t charset_size = sizeof(charset) - 1;

    for (size_t i = 0; i < length; i++) {
        int key = rand() % charset_size;
        str[i] = charset[key];
    }
    str[length] = '\0';
}

