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
typedef std::unordered_map<const char *, const char *> kernelPtrMap_t;

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

std::string nameFromDemangled(std::string demangledName)
{
    std::size_t parenthesis = demangledName.find_first_of('(');
    std::size_t anglebracket = demangledName.find_first_of('<');
    if (anglebracket != std::string::npos) {
        std::size_t returnType = demangledName.find_first_of(' ');
        if (returnType < anglebracket) {
            returnType++;
            return demangledName.substr(returnType,
                                        anglebracket - returnType);
        }
        return demangledName.substr(0, anglebracket);
    }
    if (parenthesis != std::string::npos) {
        return demangledName.substr(0, parenthesis);
    }

    return demangledName;
}

extern "C" unsigned int cudaLaunchKernel(const void  *ptr, 
                                         dim3         gridDim,
                                         dim3         blockDim,
                                         void       **args,
                                         size_t       sharedMem,
                                         void        *stream); 

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

    if (tokens[2] != "block" && tokens[2] != "thread") {
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

kernelPtrMap_t& getKernelPtrMap()
{
    static kernelPtrMap_t kernelPtrMap;
    return kernelPtrMap;
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
    kernelPtrMap_t& kernelPtrMap = getKernelPtrMap();

    kernelPtrMap[hostFun] = deviceName;

    std::string name = demangle(deviceFun);
    name = nameFromDemangled(name);
    printf("Registering %s\n", name.c_str());

    nameKernelMap[name] = hostFun;

    __cudaRegisterFunction(fatCubinHandle,
                           hostFun,
                           deviceFun,
                           deviceFun, // This parameter was repurposed.
                           thread_limit,
                           tid,
                           bid,
                           bDim,
                           gDim,
                           wSize);
}

extern "C" unsigned int rpcLaunchKernel(const void  *ptr, 
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

    // Expected format <kernelname>,<dim>,<block/thread>,<factor>,<stride>
    if (!parseConfig(kernelConfig, &config)) {
        return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
    }

    std::string nameScaled;
    nameScaled.append(config.name);
    nameScaled.append("_");
    nameScaled.append(std::to_string(config.direction));
    nameScaled.append("_");
    nameScaled.append(std::to_string(config.block ? config.factor : 1));
    nameScaled.append("_");
    nameScaled.append(std::to_string(config.block ? 1 : config.factor));
    nameScaled.append("_");
    nameScaled.append(std::to_string(config.stride));

    const nameKernelMap_t& map = getNameKernelMap();
    nameKernelMap_t::const_iterator it = map.find(nameScaled);
    if (it == map.end()) {
        printf ("RPC_ERROR: kernel not found #1 %s\n", nameScaled.c_str());
        return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
    }

    const kernelPtrMap_t& kernelPtrMap = getKernelPtrMap();
    kernelPtrMap_t::const_iterator ptrIt = kernelPtrMap.find(it->second);
    if (ptrIt == kernelPtrMap.end()) {
        printf ("RPC_ERROR: kernel not found #2 %s\n", nameScaled.c_str());
        return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
    }

    if (ptr != ptrIt->second) {
        printf ("RPC_ERROR:  kernel not found #3 %s\n", nameScaled.c_str());
        return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
    }

    if (!config.block && config.direction == 0 &&
        config.stride > (blockDim.x / config.factor)) {
        printf("RPC_ERROR: Stride parameter too big for X dimension!\n");
        return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
    }

    if (!config.block && config.direction == 1 &&
        config.stride > (blockDim.y / config.factor)) {
        printf("RPC_ERROR: Stride parameter too big for Y dimension!\n");
        return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
    }

    if (!config.block && config.direction == 2 &&
        config.stride > (blockDim.z / config.factor)) {
        printf("RPC_ERROR:  Stride parameter too big for Z dimension!\n");
        return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
    }

    dim3 *scaledDim = config.block ? &gridDim : &blockDim;
    if (config.direction == 0) {
        if (scaledDim->x / config.factor == 0) {
            return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
        }
        scaledDim->x /= config.factor;
    }
    else if (config.direction == 1) {
        if (scaledDim->y / config.factor == 0) {
            return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
        }
        scaledDim->y /= config.factor;
    }
    else {
        if (scaledDim->z / config.factor == 0) {
            return errorFallback(ptr, gridDim, blockDim, args, sharedMem, stream);
        }
        scaledDim->z /= config.factor;
    }

    return cudaLaunchKernel(it->second,
                            gridDim,
                            blockDim,
                            args,
                            sharedMem,
                            stream);
}