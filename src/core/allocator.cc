#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size) {
        ptr = nullptr;
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
    
        auto it = freeBlocks.lower_bound(size); // lower_bound返回第一个不小于size的元素迭代器
        if (it != freeBlocks.end()) {
            size_t blockAddr = it->first;
            size_t blockSize = it->second;
    
            // 从空闲块中移除
            freeBlocks.erase(it);
    
            // 如果块比需要的大，则分割并保留剩余部分为空闲
            if (blockSize > size) {
                size_t remainingAddr = blockAddr + size;
                size_t remainingSize = blockSize - size;
                freeBlocks.emplace(remainingAddr, remainingSize);
            }
    
            used += size; // 更新已用内存
            peak = std::max(peak, used);
            return blockAddr;
        } else {
            size_t addr = used;
            used += size;
            peak = std::max(used, peak);
            return addr;
        }
    }
    
    void Allocator::free(size_t addr, size_t size) {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
    
        // 尝试合并相邻的空闲块
        auto it = freeBlocks.lower_bound(addr); // 找到第一个地址不小于addr的位置
        if (it != freeBlocks.begin()) {
            --it; // 检查前一个块是否可以合并
            if (it->first + it->second == addr) { // 前一块紧邻当前释放的块
                addr = it->first;
                size += it->second;
                freeBlocks.erase(it);
            }
        }
    
        it = freeBlocks.lower_bound(addr);
        if (it != freeBlocks.end() && addr + size == it->first) { // 后一块紧邻当前释放的块
            size += it->second;
            freeBlocks.erase(it);
        }
        // 插入新的合并后的空闲块
        freeBlocks.emplace(addr, size);
    
        used -= size; // 更新已用内存
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
