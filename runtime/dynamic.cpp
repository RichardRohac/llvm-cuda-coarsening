#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <dlfcn.h>
#include <memory>
#include <cxxabi.h>
#include <stdlib.h>

#define CUDA_USES_NEW_LAUNCH 1
#define CONFIG_DELIM         ','

struct dim3 {
  unsigned x, y, z;
};

struct uint3 {
    unsigned int x, y, z;
};

typedef std::unordered_map<std::string, const char *> nameKernelMap_t;

struct coarseningConfig {
    std::string name;
    bool block;
    unsigned int factor;
    unsigned int stride;
    unsigned int direction;
};

inline std::string demangle(std::string mangledName)
{
    int status = -1;

    std::unique_ptr<char, decltype(std::free) *> result{
        abi::__cxa_demangle(mangledName.c_str(), NULL, NULL, &status),
        std::free
    };

    return (status == 0) ? result.get() : mangledName;
}

#ifdef CUDA_USES_NEW_LAUNCH
extern "C" unsigned int __cudaPushCallConfiguration(struct dim3  gridDim,
                                                    struct dim3  blockDim,
                                                    size_t       sharedMem,
                                                    void        *stream);

extern "C" unsigned int cudaLaunchKernel(const void  *ptr, 
                                         dim3         gridDim,
                                         dim3         blockDim,
                                         void       **args,
                                         size_t       sharedMem,
                                         void        *stream); 
#else
// TODO
#endif

extern "C" void __cudaRegisterFunction(void       **fatCubinHandle,
                                       const char  *hostFun,
                                       char        *deviceFun,
                                       const char  *deviceName,
                                       int          thread_limit,
                                       uint3       *tid,
                                       uint3       *bid,
                                       dim3        *bDim,
                                       dim3        *gDim,
                                       int         *wSize);

inline unsigned int errorFallback(struct dim3  gridDim,
                                  struct dim3  blockDim,
                                  size_t       sharedMem,
                                  void        *stream)
{
    std::cout << "\nRPC_CONFIG undefined or has wrong format! \
                    Running uncoarsened version!\n";

    return __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);
}

inline unsigned int errorFallback(const void  *ptr, 
                                  dim3         gridDim,
                                  dim3         blockDim,
                                  void       **args,
                                  size_t       sharedMem,
                                  void        *stream)
{
    return cudaLaunchKernel(ptr, gridDim, blockDim, args, sharedMem, stream);
}

inline bool parseConfig(char *str, coarseningConfig *result)
{
    std::istringstream ts(str);
    std::string token;

    std::vector<std::string> tokens;
    while (std::getline(ts, token, CONFIG_DELIM)) {
        tokens.push_back(token);
    }

    if (tokens.size() != 5) {
        return false;
    }

    if (tokens[1] != "block" && tokens[1] != "thread") {
        return false;
    }

    result->name = tokens[0];
    if (tokens[1] == "x") {
        result->direction = 0;
    }
    else if (tokens[1] == "y") {
        result->direction = 1;
    }
    else {
        result->direction = 2;
    }
    result->block = tokens[2] == "block";
    result->factor = atoi(tokens[3].c_str());
    result->stride = atoi(tokens[4].c_str());

    return true;
}

nameKernelMap_t& getNameKernelMap()
{
    static nameKernelMap_t nameKernelMap;
    return nameKernelMap;
}

extern "C"
const void rpcRegisterFunction(void       **fatCubinHandle,
                               const char  *hostFun,
                               char        *deviceFun,
                               const char  *deviceName,
                               int          thread_limit,
                               uint3       *tid,
                               uint3       *bid,
                               dim3        *bDim,
                               dim3        *gDim,
                               int         *wSize)
{
    nameKernelMap_t& nameKernelMap = getNameKernelMap();

    std::string name = demangle(deviceFun);
    name = name.substr(0, name.find_first_of('('));

    nameKernelMap[name] = hostFun;

    __cudaRegisterFunction(fatCubinHandle,
                           hostFun,
                           deviceFun,
                           deviceName,
                           thread_limit,
                           tid,
                           bid,
                           bDim,
                           gDim,
                           wSize);
}

extern "C" unsigned int cudaConfigureCallScaled(struct dim3  gridDim,
                                                struct dim3  blockDim,
                                                size_t       sharedMem,
                                                void        *stream)
{
    char *kernelConfig = getenv("RPC_CONFIG");
    if (!kernelConfig) {
        return errorFallback(gridDim, blockDim, sharedMem, stream);
    }

    coarseningConfig config; 

    // Expected format <kernelname>,<block/thread>,<factor>,<stride>
    if (!parseConfig(kernelConfig, &config)) {
        return errorFallback(gridDim, blockDim, sharedMem, stream);
    }

//  printf("before %i %i %i - %i %i %i\n", gridDim.x, gridDim.y, gridDim.z,
//                                         blockDim.x, blockDim.y, blockDim.z);

    dim3 *scaledDim = config.block ? &gridDim : &blockDim;
    if (config.direction == 0) {
        scaledDim->x /= config.factor;
    }
    else if (config.direction == 1) {
        scaledDim->y /= config.factor;
    }
    else {
        scaledDim->z /= config.factor;
    }

//  printf("after %i %i %i - %i %i %i\n", gridDim.x, gridDim.y, gridDim.z,
//                                        blockDim.x, blockDim.y, blockDim.z);

#ifdef CUDA_USES_NEW_LAUNCH
    return __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);
#endif
}

extern "C" unsigned int cudaLaunchDynamic(const void  *ptr, 
                                          dim3         gridDim,
                                          dim3         blockDim,
                                          void       **args,
                                          size_t       sharedMem,
                                          void        *stream)
{
    char *kernelConfig = getenv("RPC_CONFIG");
    if (!kernelConfig) {
        return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
    } 

    coarseningConfig config; 

    // Expected format <kernelname>,<block/thread>,<factor>,<stride>
    if (!parseConfig(kernelConfig, &config)) {
        return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
    }

    std::string nameScaled;
    nameScaled.append(config.name);
    nameScaled.append("_");
    nameScaled.append(std::to_string(config.block ? config.factor : 1));
    nameScaled.append("_");
    nameScaled.append(std::to_string(config.block ? 1 : config.factor));
    nameScaled.append("_");
    nameScaled.append(std::to_string(config.stride));

    const nameKernelMap_t& map = getNameKernelMap();
    nameKernelMap_t::const_iterator it = map.find(nameScaled);
    if (it == map.end()) {
        return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
    }

    return cudaLaunchKernel(it->second,
                            gridDim,
                            blockDim,
                            args,
                            sharedMem,
                            stream);
}