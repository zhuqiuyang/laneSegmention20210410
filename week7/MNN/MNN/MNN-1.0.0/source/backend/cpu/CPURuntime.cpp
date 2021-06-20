//
//  CPURuntime.cpp
//  MNN
//
//  Created by MNN on 2018/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 Ref from https://github.com/Tencent/ncnn/blob/master/src/cpu.cpp
 */
#ifdef __ANDROID__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifdef ENABLE_ARMV82

#ifdef __ANDROID__
#include <sys/auxv.h>
#include <fcntl.h>
#endif // __ANDROID__

#endif // ENABLE_ARMV82

#if __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/machine.h>
#define __IOS__ 1
#endif // TARGET_OS_IPHONE
#endif // __APPLE__


#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include "backend/cpu/CPURuntime.hpp"
#include <MNN/MNNDefine.h>

#ifdef __ANDROID__

#define BUFFER_SIZE 1024

static uint32_t getNumberOfCPU() {
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return 1;
    }
    uint32_t number = 0;
    char buffer[BUFFER_SIZE];
    while (!feof(fp)) {
        char* str = fgets(buffer, BUFFER_SIZE, fp);
        if (!str) {
            break;
        }
        if (memcmp(buffer, "processor", 9) == 0) {
            number++;
        }
    }
    fclose(fp);
    if (number < 1) {
        number = 1;
    }
    return number;
}

static int getCPUMaxFreqKHz(int cpuID) {
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuID);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuID);
        fp = fopen(path, "rb");
        if (!fp) {
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuID);
            fp = fopen(path, "rb");
            if (!fp) {
                return -1;
            }
            int maxfrequency = -1;
            fscanf(fp, "%d", &maxfrequency);
            fclose(fp);
            return maxfrequency;
        }
    }
    int maxfrequency = 0;
    while (!feof(fp)) {
        int frequency = 0;
        int history   = fscanf(fp, "%d %*d", &frequency);
        if (history != 1) {
            break;
        }
        if (frequency > maxfrequency) {
            maxfrequency = frequency;
        }
    }
    fclose(fp);
    return maxfrequency;
}

static int sortCPUIDByMaxFrequency(std::vector<int>& cpuIDs, int* littleClusterOffset) {
    const int cpuNumbers = cpuIDs.size();
    *littleClusterOffset = 0;
    if (cpuNumbers == 0) {
        return 0;
    }
    std::vector<int> cpusFrequency;
    cpusFrequency.resize(cpuNumbers);
    for (int i = 0; i < cpuNumbers; ++i) {
        int frequency    = getCPUMaxFreqKHz(i);
        cpuIDs[i]        = i;
        cpusFrequency[i] = frequency;
        // MNN_PRINT("cpu fre: %d, %d\n", i, frequency);
    }
    for (int i = 0; i < cpuNumbers; ++i) {
        for (int j = i + 1; j < cpuNumbers; ++j) {
            if (cpusFrequency[i] < cpusFrequency[j]) {
                // id
                int temp  = cpuIDs[i];
                cpuIDs[i] = cpuIDs[j];
                cpuIDs[j] = temp;
                // frequency
                temp             = cpusFrequency[i];
                cpusFrequency[i] = cpusFrequency[j];
                cpusFrequency[j] = temp;
            }
        }
    }
    int midMaxFrequency = (cpusFrequency.front() + cpusFrequency.back()) / 2;
    if (midMaxFrequency == cpusFrequency.back()) {
        return 0;
    }
    for (int i = 0; i < cpuNumbers; ++i) {
        if (cpusFrequency[i] < midMaxFrequency) {
            *littleClusterOffset = i;
            break;
        }
    }
    return 0;
}

static int setSchedAffinity(const std::vector<int>& cpuIDs) {
#define CPU_SETSIZE 1024
#define __NCPUBITS (8 * sizeof(unsigned long))
    typedef struct {
        unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
    } cpu_set_t;

#define CPU_SET(cpu, cpusetp) ((cpusetp)->__bits[(cpu) / __NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))

    // set affinity for thread
#ifdef __GLIBC__
    pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
    pid_t pid = getpid();
#else
    pid_t pid = gettid();
#endif
#endif
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i = 0; i < (int)cpuIDs.size(); i++) {
        CPU_SET(cpuIDs[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret) {
        MNN_PRINT("syscall error %d\n", syscallret);
        return -1;
    }

    return 0;
}

#endif // arch

int MNNSetCPUThreadsMode(MNNCPUThreadsMode mode) {
#ifdef __ANDROID__
    auto numberOfCPUs = getNumberOfCPU();
    if (mode == MNN_CPU_MODE_DEFAULT) {
        return 0;
    }
    static std::vector<int> sortedCPUIDs;
    static int littleClusterOffset = 0;
    if (sortedCPUIDs.empty()) {
        sortedCPUIDs.resize(numberOfCPUs);
        for (int i = 0; i < numberOfCPUs; ++i) {
            sortedCPUIDs[i] = i;
        }
        sortCPUIDByMaxFrequency(sortedCPUIDs, &littleClusterOffset);
    }

    if (littleClusterOffset == 0 && mode != MNN_CPU_MODE_POWER_FRI) {
        MNN_PRINT("This CPU Arch Do NOT support for setting cpu thread mode\n");
    }
    std::vector<int> cpuAttachIDs;
    switch (mode) {
        case MNN_CPU_MODE_POWER_FRI:
            cpuAttachIDs = sortedCPUIDs;
            break;
        case MNN_CPU_MODE_LITTLE:
            cpuAttachIDs = std::vector<int>(sortedCPUIDs.begin() + littleClusterOffset, sortedCPUIDs.end());
            break;
        case MNN_CPU_MODE_BIG:
            cpuAttachIDs = std::vector<int>(sortedCPUIDs.begin(), sortedCPUIDs.begin() + littleClusterOffset);
            break;
        default:
            cpuAttachIDs = sortedCPUIDs;
            break;
    }

#ifdef _OPENMP
    const int threadsNumber = cpuAttachIDs.size();
    omp_set_num_threads(threadsNumber);
    std::vector<int> result(threadsNumber, 0);
#pragma omp parallel for
    for (int i = 0; i < threadsNumber; ++i) {
        result[i] = setSchedAffinity(cpuAttachIDs);
    }
    for (int i = 0; i < threadsNumber; ++i) {
        if (result[i] != 0) {
            return -1;
        }
    }
#else
    int res   = setSchedAffinity(cpuAttachIDs);
    if (res != 0) {
        return -1;
    }
#endif // _OPENMP
    return 0;
#elif __IOS__
    return -1;
#else
    return -1;
#endif // arch
}
float MNNGetCPUFlops(uint32_t number) {
    float flops = 2048.0f;
#ifdef __ANDROID__
    auto numberOfCPUs = getNumberOfCPU();
    if (0 == numberOfCPUs) {
        return flops;
    }
    std::vector<int> freqs;
    freqs.resize(numberOfCPUs);
    for (int i = 0; i < numberOfCPUs; ++i) {
        freqs[i]    = getCPUMaxFreqKHz(i);
    }
    std::sort(freqs.rbegin(), freqs.rend());
    number = std::min(number, numberOfCPUs);
    flops = 0.0f;
    for (uint32_t i=0; i<number; ++i) {
        flops += (float)freqs[i] / 1024.0f;
    }
#endif
    return flops;
}


// cpuinfo
// Reference from: https://github.com/pytorch/cpuinfo

#ifdef ENABLE_ARMV82

#ifdef __ANDROID__

#define CPUINFO_HARDWARE_VALUE_MAX 64

#define CPUINFO_ARM_MIDR_IMPLEMENTER_MASK  UINT32_C(0xFF000000)
#define CPUINFO_ARM_MIDR_VARIANT_MASK      UINT32_C(0x00F00000)
#define CPUINFO_ARM_MIDR_ARCHITECTURE_MASK UINT32_C(0x000F0000)
#define CPUINFO_ARM_MIDR_PART_MASK         UINT32_C(0x0000FFF0)
#define CPUINFO_ARM_MIDR_REVISION_MASK     UINT32_C(0x0000000F)

#define CPUINFO_ARM_LINUX_VALID_ARCHITECTURE  UINT32_C(0x00010000)
#define CPUINFO_ARM_LINUX_VALID_IMPLEMENTER   UINT32_C(0x00020000)
#define CPUINFO_ARM_LINUX_VALID_VARIANT      UINT32_C(0x00040000)
#define CPUINFO_LINUX_FLAG_VALID              UINT32_C(0x00001000)
#define CPUINFO_ARM_LINUX_VALID_MIDR          UINT32_C(0x003F0000)
#define CPUINFO_ARM_LINUX_VALID_PART          UINT32_C(0x00080000)
#define CPUINFO_ARM_LINUX_VALID_PROCESSOR     UINT32_C(0x00200000)
#define CPUINFO_ARM_LINUX_VALID_REVISION      UINT32_C(0x00100000)

#define CPUINFO_ARM_MIDR_IMPLEMENTER_OFFSET  24
#define CPUINFO_ARM_MIDR_VARIANT_OFFSET      20
#define CPUINFO_ARM_MIDR_ARCHITECTURE_OFFSET 16
#define CPUINFO_ARM_MIDR_PART_OFFSET          4
#define CPUINFO_ARM_MIDR_REVISION_OFFSET      0

#define CPUINFO_ARM_LINUX_FEATURE_FPHP     UINT32_C(0x00000200)
#define CPUINFO_ARM_LINUX_FEATURE_ASIMDHP  UINT32_C(0x00000400)
#define CPUINFO_ARM_LINUX_FEATURE_ASIMDDP  UINT32_C(0x00100000)

struct cpuinfo_arm_linux_processor{
    uint32_t architecture_version;
    // Main ID Register value
    uint32_t midr;
    
    uint32_t max_frequency;
    uint32_t min_frequency;
    
    uint32_t system_processor_id;
    uint32_t flags;
};

struct proc_cpuinfo_parser_state {
    char* hardware;
    uint32_t processor_index;
    uint32_t max_processors_count;
    struct cpuinfo_arm_linux_processor* processors;
    struct cpuinfo_arm_linux_processor dummy_processor;
};


typedef bool (*cpuinfo_line_callback)(const char*, const char*, void*, uint64_t);

inline static uint32_t midr_set_implementer(uint32_t midr, uint32_t implementer) {
    return (midr & ~CPUINFO_ARM_MIDR_IMPLEMENTER_MASK) |
        ((implementer << CPUINFO_ARM_MIDR_IMPLEMENTER_OFFSET) & CPUINFO_ARM_MIDR_IMPLEMENTER_MASK);
}

inline static uint32_t midr_set_architecture(uint32_t midr, uint32_t architecture) {
    return (midr & ~CPUINFO_ARM_MIDR_ARCHITECTURE_MASK) |
        ((architecture << CPUINFO_ARM_MIDR_ARCHITECTURE_OFFSET) & CPUINFO_ARM_MIDR_ARCHITECTURE_MASK);
}

inline static uint32_t midr_set_part(uint32_t midr, uint32_t part) {
    return (midr & ~CPUINFO_ARM_MIDR_PART_MASK) |
        ((part << CPUINFO_ARM_MIDR_PART_OFFSET) & CPUINFO_ARM_MIDR_PART_MASK);
}

inline static uint32_t midr_set_revision(uint32_t midr, uint32_t revision) {
    return (midr & ~CPUINFO_ARM_MIDR_REVISION_MASK) |
        ((revision << CPUINFO_ARM_MIDR_REVISION_OFFSET) & CPUINFO_ARM_MIDR_REVISION_MASK);
}

inline static uint32_t midr_set_variant(uint32_t midr, uint32_t variant) {
    return (midr & ~CPUINFO_ARM_MIDR_VARIANT_MASK) |
        ((variant << CPUINFO_ARM_MIDR_VARIANT_OFFSET) & CPUINFO_ARM_MIDR_VARIANT_MASK);
}

uint32_t cpuinfo_arm_linux_hwcap_from_getauxval(void){
    return (uint32_t) getauxval(AT_HWCAP);
}

static inline bool bitmask_all(uint32_t bitfield, uint32_t mask) {
    return (bitfield & mask) == mask;
}

static void parse_cpu_part(
    const char* cpu_part_start,
    const char* cpu_part_end,
    struct cpuinfo_arm_linux_processor* processor)
{
    const size_t cpu_part_length = (size_t) (cpu_part_end - cpu_part_start);

    /*
     * CPU part should contain hex prefix (0x) and one to three hex digits.
     * I have never seen less than three digits as a value of this field,
     * but I don't think it is impossible to see such values in future.
     * Value can not contain more than three hex digits since
     * Main ID Register (MIDR) assigns only a 12-bit value for CPU part.
     */
    if (cpu_part_length < 3 || cpu_part_length > 5) {
        MNN_PRINT("CPU part %.*s in /proc/cpuinfo is ignored due to unexpected length (%zu)",
            (int) cpu_part_length, cpu_part_start, cpu_part_length);
        return;
    }

    /* Verify the presence of hex prefix */
    if (cpu_part_start[0] != '0' || cpu_part_start[1] != 'x') {
        MNN_PRINT("CPU part %.*s in /proc/cpuinfo is ignored due to lack of 0x prefix",
            (int) cpu_part_length, cpu_part_start);
        return;
    }

    /* Verify that characters after hex prefix are hexadecimal digits and decode them */
    uint32_t cpu_part = 0;
    for (const char* digit_ptr = cpu_part_start + 2; digit_ptr != cpu_part_end; digit_ptr++) {
        const char digit_char = *digit_ptr;
        uint32_t digit;
        if (digit_char >= '0' && digit_char <= '9') {
            digit = digit_char - '0';
        } else if ((uint32_t) (digit_char - 'A') < 6) {
            digit = 10 + (digit_char - 'A');
        } else if ((uint32_t) (digit_char - 'a') < 6) {
            digit = 10 + (digit_char - 'a');
        } else {
            MNN_PRINT("CPU part %.*s in /proc/cpuinfo is ignored due to unexpected non-hex character %c at offset %zu",
                (int) cpu_part_length, cpu_part_start, digit_char, (size_t) (digit_ptr - cpu_part_start));
            return;
        }
        cpu_part = cpu_part * 16 + digit;
    }

    processor->midr = midr_set_part(processor->midr, cpu_part);
    processor->flags |= CPUINFO_ARM_LINUX_VALID_PART | CPUINFO_ARM_LINUX_VALID_PROCESSOR;
}

static void parse_cpu_revision(
    const char* cpu_revision_start,
    const char* cpu_revision_end,
    struct cpuinfo_arm_linux_processor* processor)
{
    uint32_t cpu_revision = 0;
    for (const char* digit_ptr = cpu_revision_start; digit_ptr != cpu_revision_end; digit_ptr++) {
        const uint32_t digit = (uint32_t) (*digit_ptr - '0');

        /* Verify that the character in CPU revision is a decimal digit */
        if (digit >= 10) {
            MNN_PRINT("CPU revision %.*s in /proc/cpuinfo is ignored due to unexpected non-digit character '%c' at offset %zu",
                (int) (cpu_revision_end - cpu_revision_start), cpu_revision_start,
                *digit_ptr, (size_t) (digit_ptr - cpu_revision_start));
            return;
        }

        cpu_revision = cpu_revision * 10 + digit;
    }

    processor->midr = midr_set_revision(processor->midr, cpu_revision);
    processor->flags |= CPUINFO_ARM_LINUX_VALID_REVISION | CPUINFO_ARM_LINUX_VALID_PROCESSOR;
}

static void parse_cpu_architecture(
    const char* cpu_architecture_start,
    const char* cpu_architecture_end,
    struct cpuinfo_arm_linux_processor* processor)
{
    const size_t cpu_architecture_length = (size_t) (cpu_architecture_end - cpu_architecture_start);
    /* Early AArch64 kernels report "CPU architecture: AArch64" instead of a numeric value 8 */
    if (cpu_architecture_length == 7) {
        if (memcmp(cpu_architecture_start, "AArch64", cpu_architecture_length) == 0) {
            processor->midr = midr_set_architecture(processor->midr, UINT32_C(0xF));
            processor->architecture_version = 8;
            processor->flags |= CPUINFO_ARM_LINUX_VALID_ARCHITECTURE | CPUINFO_ARM_LINUX_VALID_PROCESSOR;
            return;
        }
    }


    uint32_t architecture = 0;
    const char* cpu_architecture_ptr = cpu_architecture_start;
    for (; cpu_architecture_ptr != cpu_architecture_end; cpu_architecture_ptr++) {
        const uint32_t digit = (*cpu_architecture_ptr) - '0';

        /* Verify that CPU architecture is a decimal number */
        if (digit >= 10) {
            break;
        }

        architecture = architecture * 10 + digit;
    }

    if (cpu_architecture_ptr == cpu_architecture_start) {
        MNN_PRINT("CPU architecture %.*s in /proc/cpuinfo is ignored due to non-digit at the beginning of the string",
            (int) cpu_architecture_length, cpu_architecture_start);
    } else {
        if (architecture != 0) {
            processor->architecture_version = architecture;
            processor->flags |= CPUINFO_ARM_LINUX_VALID_ARCHITECTURE | CPUINFO_ARM_LINUX_VALID_PROCESSOR;

            for (; cpu_architecture_ptr != cpu_architecture_end; cpu_architecture_ptr++) {
                const char feature = *cpu_architecture_ptr;
                switch (feature) {
                    case ' ':
                    case '\t':
                        /* Ignore whitespace at the end */
                        break;
                    default:
                        MNN_PRINT("skipped unknown architectural feature '%c' for ARMv%u",
                            feature, architecture);
                        break;
                }
            }
        } else {
            MNN_PRINT("CPU architecture %.*s in /proc/cpuinfo is ignored due to invalid value (0)",
                (int) cpu_architecture_length, cpu_architecture_start);
        }
    }

    uint32_t midr_architecture = UINT32_C(0xF);
    processor->midr = midr_set_architecture(processor->midr, midr_architecture);
}

static uint32_t parse_processor_number(
    const char* processor_start,
    const char* processor_end)
{
    const size_t processor_length = (size_t) (processor_end - processor_start);

    if (processor_length == 0) {
        MNN_PRINT("Processor number in /proc/cpuinfo is ignored: string is empty");
        return 0;
    }

    uint32_t processor_number = 0;
    for (const char* digit_ptr = processor_start; digit_ptr != processor_end; digit_ptr++) {
        const uint32_t digit = (uint32_t) (*digit_ptr - '0');
        if (digit > 10) {
            MNN_PRINT("non-decimal suffix %.*s in /proc/cpuinfo processor number is ignored",
                (int) (processor_end - digit_ptr), digit_ptr);
            break;
        }

        processor_number = processor_number * 10 + digit;
    }

    return processor_number;
}

static void parse_cpu_variant(
    const char* cpu_variant_start,
    const char* cpu_variant_end,
    struct cpuinfo_arm_linux_processor* processor)
{
    const size_t cpu_variant_length = cpu_variant_end - cpu_variant_start;

    /*
     * Value should contain hex prefix (0x) and one hex digit.
     * Value can not contain more than one hex digits since
     * Main ID Register (MIDR) assigns only a 4-bit value for CPU variant.
     */
    if (cpu_variant_length != 3) {
        MNN_PRINT("CPU variant %.*s in /proc/cpuinfo is ignored due to unexpected length (%zu)",
            (int) cpu_variant_length, cpu_variant_start, cpu_variant_length);
        return;
    }

    /* Skip if there is no hex prefix (0x) */
    if (cpu_variant_start[0] != '0' || cpu_variant_start[1] != 'x') {
        MNN_PRINT("CPU variant %.*s in /proc/cpuinfo is ignored due to lack of 0x prefix",
            (int) cpu_variant_length, cpu_variant_start);
        return;
    }

    /* Check if the value after hex prefix is indeed a hex digit and decode it. */
    const char digit_char = cpu_variant_start[2];
    uint32_t cpu_variant;
    if ((uint32_t) (digit_char - '0') < 10) {
        cpu_variant = (uint32_t) (digit_char - '0');
    } else if ((uint32_t) (digit_char - 'A') < 6) {
        cpu_variant = 10 + (uint32_t) (digit_char - 'A');
    } else if ((uint32_t) (digit_char - 'a') < 6) {
        cpu_variant = 10 + (uint32_t) (digit_char - 'a');
    } else {
        MNN_PRINT("CPU variant %.*s in /proc/cpuinfo is ignored due to unexpected non-hex character '%c'",
            (int) cpu_variant_length, cpu_variant_start, digit_char);
        return;
    }

    processor->midr = midr_set_variant(processor->midr, cpu_variant);
    processor->flags |= CPUINFO_ARM_LINUX_VALID_VARIANT | CPUINFO_ARM_LINUX_VALID_PROCESSOR;
}

static void parse_cpu_implementer(
    const char* cpu_implementer_start,
    const char* cpu_implementer_end,
    struct cpuinfo_arm_linux_processor* processor)
{
    const size_t cpu_implementer_length = cpu_implementer_end - cpu_implementer_start;

    /*
     * Value should contain hex prefix (0x) and one or two hex digits.
     * I have never seen single hex digit as a value of this field,
     * but I don't think it is impossible in future.
     * Value can not contain more than two hex digits since
     * Main ID Register (MIDR) assigns only an 8-bit value for CPU implementer.
     */
    switch (cpu_implementer_length) {
        case 3:
        case 4:
            break;
        default:
        MNN_PRINT("CPU implementer %.*s in /proc/cpuinfo is ignored due to unexpected length (%zu)",
            (int) cpu_implementer_length, cpu_implementer_start, cpu_implementer_length);
        return;
    }

    /* Verify the presence of hex prefix */
    if (cpu_implementer_start[0] != '0' || cpu_implementer_start[1] != 'x') {
        MNN_PRINT("CPU implementer %.*s in /proc/cpuinfo is ignored due to lack of 0x prefix",
            (int) cpu_implementer_length, cpu_implementer_start);
        return;
    }

    /* Verify that characters after hex prefix are hexadecimal digits and decode them */
    uint32_t cpu_implementer = 0;
    for (const char* digit_ptr = cpu_implementer_start + 2; digit_ptr != cpu_implementer_end; digit_ptr++) {
        const char digit_char = *digit_ptr;
        uint32_t digit;
        if (digit_char >= '0' && digit_char <= '9') {
            digit = digit_char - '0';
        } else if ((uint32_t) (digit_char - 'A') < 6) {
            digit = 10 + (digit_char - 'A');
        } else if ((uint32_t) (digit_char - 'a') < 6) {
            digit = 10 + (digit_char - 'a');
        } else {
            MNN_PRINT("CPU implementer %.*s in /proc/cpuinfo is ignored due to unexpected non-hex character '%c' at offset %zu",
                (int) cpu_implementer_length, cpu_implementer_start, digit_char, (size_t) (digit_ptr - cpu_implementer_start));
            return;
        }
        cpu_implementer = cpu_implementer * 16 + digit;
    }

    processor->midr = midr_set_implementer(processor->midr, cpu_implementer);
    processor->flags |= CPUINFO_ARM_LINUX_VALID_IMPLEMENTER | CPUINFO_ARM_LINUX_VALID_PROCESSOR;
}

static bool parse_line(
    const char* line_start,
    const char* line_end,
    struct proc_cpuinfo_parser_state* state,
    uint64_t line_number)
{
    /* Empty line. Skip. */
    if (line_start == line_end) {
        return true;
    }
    
    /* Search for ':' on the line. */
    const char* separator = line_start;
    for (; separator != line_end; separator++) {
        if (*separator == ':') {
            break;
        }
    }
    /* Skip line if no ':' separator was found. */
    if (separator == line_end) {
        MNN_PRINT("Line %.*s in /proc/cpuinfo is ignored: key/value separator ':' not found",
            (int) (line_end - line_start), line_start);
        return true;
    }

    /* Skip trailing spaces in key part. */
    const char* key_end = separator;
    for (; key_end != line_start; key_end--) {
        if (key_end[-1] != ' ' && key_end[-1] != '\t') {
            break;
        }
    }
    /* Skip line if key contains nothing but spaces. */
    if (key_end == line_start) {
        MNN_PRINT("Line %.*s in /proc/cpuinfo is ignored: key contains only spaces",
            (int) (line_end - line_start), line_start);
        return true;
    }

    /* Skip leading spaces in value part. */
    const char* value_start = separator + 1;
    for (; value_start != line_end; value_start++) {
        if (*value_start != ' ') {
            break;
        }
    }
    /* Value part contains nothing but spaces. Skip line. */
    if (value_start == line_end) {
        MNN_PRINT("Line %.*s in /proc/cpuinfo is ignored: value contains only spaces",
            (int) (line_end - line_start), line_start);
        return true;
    }

    /* Skip trailing spaces in value part (if any) */
    const char* value_end = line_end;
    for (; value_end != value_start; value_end--) {
        if (value_end[-1] != ' ') {
            break;
        }
    }

    const uint32_t processor_index      = state->processor_index;
    const uint32_t max_processors_count = state->max_processors_count;
    struct cpuinfo_arm_linux_processor* processors = state->processors;
    struct cpuinfo_arm_linux_processor* processor  = &state->dummy_processor;
    if (processor_index < max_processors_count) {
        processor = &processors[processor_index];
    }

    const size_t key_length = key_end - line_start;
    switch (key_length) {
        case 6:
            if (memcmp(line_start, "Serial", key_length) == 0) {
                /* Usually contains just zeros, useless */
            } else {
                MNN_PRINT("unknown /proc/cpuinfo key: %.*s\n", (int) key_length, line_start);
            }
            break;
        case 8:
            if (memcmp(line_start, "CPU part", key_length) == 0) {
                parse_cpu_part(value_start, value_end, processor);
            } else if (memcmp(line_start, "Features", key_length) == 0) {
                /* parse_features(value_start, value_end, processor); */
            } else if (memcmp(line_start, "BogoMIPS", key_length) == 0) {
                /* BogoMIPS is useless, don't parse */
            } else if (memcmp(line_start, "Hardware", key_length) == 0) {
                size_t value_length = value_end - value_start;
                if (value_length > CPUINFO_HARDWARE_VALUE_MAX) {
                    MNN_PRINT(
                        "length of Hardware value \"%.*s\" in /proc/cpuinfo exceeds limit (%d): truncating to the limit\n",
                        (int) value_length, value_start, CPUINFO_HARDWARE_VALUE_MAX);
                    value_length = CPUINFO_HARDWARE_VALUE_MAX;
                } else {
                    state->hardware[value_length] = '\0';
                }
                memcpy(state->hardware, value_start, value_length);
                MNN_PRINT("parsed /proc/cpuinfo Hardware = \"%.*s\"\n", (int) value_length, value_start);
            } else if (memcmp(line_start, "Revision", key_length) == 0) {
                /* Board revision, no use for now */
            } else {
                MNN_PRINT("unknown /proc/cpuinfo key: %.*s\n", (int) key_length, line_start);
            }
            break;
        case 9:
            if (memcmp(line_start, "processor", key_length) == 0) {
                const uint32_t new_processor_index = parse_processor_number(value_start, value_end);
                if (new_processor_index < processor_index) {
                    /* Strange: decreasing processor number */
                    MNN_PRINT(
                        "unexpectedly low processor number %u following processor %u in /proc/cpuinfo\n",
                        new_processor_index, processor_index);
                } else if (new_processor_index > processor_index + 1) {
                    /* Strange, but common: skipped processor $(processor_index + 1) */
                    MNN_PRINT(
                        "unexpectedly high processor number %u following processor %u in /proc/cpuinfo\n",
                        new_processor_index, processor_index);
                }
                if (new_processor_index < max_processors_count) {
                    /* Record that the processor was mentioned in /proc/cpuinfo */
                    processors[new_processor_index].flags |= CPUINFO_ARM_LINUX_VALID_PROCESSOR;
                } else {
                    /* Log and ignore processor */
                    MNN_PRINT("processor %u in /proc/cpuinfo is ignored: index exceeds system limit %u\n",
                        new_processor_index, max_processors_count - 1);
                }
                state->processor_index = new_processor_index;
                return true;
            } else if (memcmp(line_start, "Processor", key_length) == 0) {
                /* TODO: parse to fix misreported architecture, similar to Android's cpufeatures */
            } else {
                MNN_PRINT("unknown /proc/cpuinfo key: %.*s\n", (int) key_length, line_start);
            }
            break;
        case 11:
            if (memcmp(line_start, "CPU variant", key_length) == 0) {
                parse_cpu_variant(value_start, value_end, processor);
            } else {
                MNN_PRINT("unknown /proc/cpuinfo key: %.*s\n", (int) key_length, line_start);
            }
            break;
        case 12:
            if (memcmp(line_start, "CPU revision", key_length) == 0) {
                parse_cpu_revision(value_start, value_end, processor);
            } else {
                MNN_PRINT("unknown /proc/cpuinfo key: %.*s\n", (int) key_length, line_start);
            }
            break;
        case 15:
            if (memcmp(line_start, "CPU implementer", key_length) == 0) {
                parse_cpu_implementer(value_start, value_end, processor);
            } else if (memcmp(line_start, "CPU implementor", key_length) == 0) {
                parse_cpu_implementer(value_start, value_end, processor);
            } else {
                MNN_PRINT("unknown /proc/cpuinfo key: %.*s\n", (int) key_length, line_start);
            }
            break;
        case 16:
            if (memcmp(line_start, "CPU architecture", key_length) == 0) {
                parse_cpu_architecture(value_start, value_end, processor);
            } else {
                MNN_PRINT("unknown /proc/cpuinfo key: %.*s\n", (int) key_length, line_start);
            }
            break;
        default:
            MNN_PRINT("unknown /proc/cpuinfo key: %.*s\n", (int) key_length, line_start);

    }
    return true;
}

bool cpuinfo_linux_parse_multiline_file(const char* filename, size_t buffer_size, cpuinfo_line_callback callback, void* context)
{
#define RETIEMENT if (file != -1) { \
    close(file); \
    file = -1; \
} \
return false;
    
    int file = -1;
    bool status = false;
    char* buffer = (char*) alloca(buffer_size);


    file = open(filename, O_RDONLY);
    if (file == -1) {
        MNN_PRINT("failed to open %s\n", filename);
        RETIEMENT
    }

    /* Only used for error reporting */
    size_t position = 0;
    uint64_t line_number = 1;
    const char* buffer_end = &buffer[buffer_size];
    char* data_start = buffer;
    ssize_t bytes_read;
    do {
        bytes_read = read(file, data_start, (size_t) (buffer_end - data_start));
        if (bytes_read < 0) {
            MNN_PRINT("failed to read file %s at position %zu\n",
                filename, position);
            RETIEMENT
        }

        position += (size_t) bytes_read;
        const char* data_end = data_start + (size_t) bytes_read;
        const char* line_start = buffer;

        if (bytes_read == 0) {
            /* No more data in the file: process the remaining text in the buffer as a single entry */
            const char* line_end = data_end;
            if (!callback(line_start, line_end, context, line_number)) {
                RETIEMENT
            }
        } else {
            const char* line_end;
            do {
                /* Find the end of the entry, as indicated by newline character ('\n') */
                for (line_end = line_start; line_end != data_end; line_end++) {
                    if (*line_end == '\n') {
                        break;
                    }
                }

                /*
                 * If we located separator at the end of the entry, parse it.
                 * Otherwise, there may be more data at the end; read the file once again.
                 */
                if (line_end != data_end) {
                    if (!callback(line_start, line_end, context, line_number++)) {
                        RETIEMENT
                    }
                    line_start = line_end + 1;
                }
            } while (line_end != data_end);

            /* Move remaining partial line data at the end to the beginning of the buffer */
            const size_t line_length = (size_t) (line_end - line_start);
            memmove(buffer, line_start, line_length);
            data_start = &buffer[line_length];
        }
    } while (bytes_read != 0);

    /* Commit */
    status = true;

//cleanup:
//    if (file != -1) {
//        close(file);
//        file = -1;
//    }
    return status;
}

bool cpuinfo_arm_linux_parse_proc_cpuinfo(char* hardware, uint32_t max_processors_count, struct cpuinfo_arm_linux_processor* processors){
    struct proc_cpuinfo_parser_state state = {
        .hardware = hardware,
        .processor_index = 0,
        .max_processors_count = max_processors_count,
        .processors = processors,
    };
    
    return cpuinfo_linux_parse_multiline_file("/proc/cpuinfo", BUFFER_SIZE, (cpuinfo_line_callback)parse_line, &state);
}

#endif // __ANDROID__

#if defined(__IOS__) && defined(__aarch64__)

static uint32_t get_sys_info_by_name(const char* type_specifier) {
    size_t size = 0;
    uint32_t result = 0;
    if (sysctlbyname(type_specifier, NULL, &size, NULL, 0) != 0) {
        MNN_PRINT("sysctlbyname(\"%s\") failed\n", type_specifier);
    } else if (size == sizeof(uint32_t)) {
        sysctlbyname(type_specifier, &result, &size, NULL, 0);
        MNN_PRINT("%s: %u , size = %lu\n", type_specifier, result, size);
    } else {
        MNN_PRINT("sysctl does not support non-integer lookup for (\"%s\")\n", type_specifier);
    }
    return result;
}



#endif // iOS


void cpuinfo_arm_init(struct cpuinfo_arm_isa* cpuinfo_isa){
    memset(cpuinfo_isa, 0, sizeof(struct cpuinfo_arm_isa));
    
    // android
#ifdef __ANDROID__
    struct cpuinfo_arm_linux_processor* arm_linux_processors = NULL;
    const uint32_t processors_count = getNumberOfCPU();

    char proc_cpuinfo_hardware[CPUINFO_HARDWARE_VALUE_MAX] = { 0 };
    
    arm_linux_processors = static_cast<struct cpuinfo_arm_linux_processor*>(calloc(processors_count, sizeof(struct cpuinfo_arm_linux_processor)));
    if(arm_linux_processors == NULL){
        MNN_PRINT(
            "failed to allocate %zu bytes for descriptions of %u ARM logical processors",
            processors_count * sizeof(struct cpuinfo_arm_linux_processor),
            processors_count);
        return;
    }
    
    if(!cpuinfo_arm_linux_parse_proc_cpuinfo(proc_cpuinfo_hardware, processors_count, arm_linux_processors)){
        MNN_PRINT("failed to parse processor information from /proc/cpuinfo\n");
        return;
    }
    
    uint32_t valid_processor_mask = 0;
    for(uint32_t i = 0; i < processors_count; i++){
        if(bitmask_all(arm_linux_processors[i].flags, valid_processor_mask)){
            arm_linux_processors[i].flags |= CPUINFO_LINUX_FLAG_VALID;
        }
    }

    uint32_t valid_processors = 0, last_midr = 0;
    for (uint32_t i = 0; i < processors_count; i++) {
        arm_linux_processors[i].system_processor_id = i;
        if (bitmask_all(arm_linux_processors[i].flags, CPUINFO_LINUX_FLAG_VALID)){
            valid_processors += 1;
            if (bitmask_all(arm_linux_processors[i].flags, CPUINFO_ARM_LINUX_VALID_MIDR)) {
                last_midr = arm_linux_processors[i].midr;
            }
        }
    }
    
    const uint32_t isa_features = cpuinfo_arm_linux_hwcap_from_getauxval();
    
    switch (last_midr & (CPUINFO_ARM_MIDR_IMPLEMENTER_MASK | CPUINFO_ARM_MIDR_PART_MASK)) {
        case UINT32_C(0x51008040): /* Kryo 485 Gold (Cortex-A76) */
            cpuinfo_isa->dot = true;
            break;
        default:
            if (isa_features & CPUINFO_ARM_LINUX_FEATURE_ASIMDDP) {
                cpuinfo_isa->dot = true;
            }
            // TODO, whitelist, ex: hisilicon_kirin 980...
            break;
    }
    
    const uint32_t fp16arith_mask = CPUINFO_ARM_LINUX_FEATURE_FPHP | CPUINFO_ARM_LINUX_FEATURE_ASIMDHP;
    if((isa_features & fp16arith_mask) == fp16arith_mask){
        // TODO, blacklist, ex: samsubg_exynos 9810
        cpuinfo_isa->fp16arith = true;
    }
    
#endif // #ifdef __ANDROID__
    
    // iOS
#if defined(__IOS__) && defined(__aarch64__)

// A11
#ifndef CPUFAMILY_ARM_MONSOON_MISTRAL
#define CPUFAMILY_ARM_MONSOON_MISTRAL   0xe81e7ef6
#endif
// A12
#ifndef CPUFAMILY_ARM_VORTEX_TEMPEST
#define CPUFAMILY_ARM_VORTEX_TEMPEST    0x07d34b9f
#endif
// A13
#ifndef CPUFAMILY_ARM_LIGHTNING_THUNDER
#define CPUFAMILY_ARM_LIGHTNING_THUNDER 0x462504d2
#endif

    const uint32_t cpu_family = get_sys_info_by_name("hw.cpufamily");
    // const uint32_t cpu_type = get_sys_info_by_name("hw.cputype");
    // const uint32_t cpu_subtype = get_sys_info_by_name("hw.cpusubtype");
    
    cpuinfo_isa->fp16arith = cpu_family == CPUFAMILY_ARM_MONSOON_MISTRAL || cpu_family == CPUFAMILY_ARM_VORTEX_TEMPEST || cpu_family == CPUFAMILY_ARM_LIGHTNING_THUNDER;
    
    cpuinfo_isa->dot = cpu_family == CPUFAMILY_ARM_LIGHTNING_THUNDER;
    
#endif // iOS
}


#endif // ENABLE_ARMV82