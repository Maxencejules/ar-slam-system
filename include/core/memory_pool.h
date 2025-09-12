#pragma once
#include <vector>
#include <memory>
#include <cstddef>
#include <algorithm>

namespace ar_slam {

    template<typename T>
    class MemoryPool {
    private:
        size_t max_size_;
        std::vector<T*> allocated_items_;

    public:
        explicit MemoryPool(size_t max_bytes = 256 * 1024 * 1024)
            : max_size_(max_bytes) {
            allocated_items_.reserve(100);
        }

        ~MemoryPool() {
            // Don't delete items here - user is responsible for calling destructors
            for (auto* item : allocated_items_) {
                ::operator delete(item);  // Just free memory, don't call destructor
            }
        }

        T* allocate() {
            // Allocate memory without constructing
            void* mem = ::operator new(sizeof(T));
            T* ptr = static_cast<T*>(mem);
            allocated_items_.push_back(ptr);
            return ptr;
        }

        void deallocate(T* ptr) {
            auto it = std::find(allocated_items_.begin(), allocated_items_.end(), ptr);
            if (it != allocated_items_.end()) {
                ::operator delete(ptr);  // Just free memory
                allocated_items_.erase(it);
            }
        }

        size_t get_usage() const {
            return allocated_items_.size() * sizeof(T);
        }
    };

} // namespace ar_slam