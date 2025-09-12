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
            for (auto* item : allocated_items_) {
                delete item;
            }
        }

        T* allocate() {
            T* new_item = new T();
            allocated_items_.push_back(new_item);
            return new_item;
        }

        void deallocate(T* ptr) {
            auto it = std::find(allocated_items_.begin(), allocated_items_.end(), ptr);
            if (it != allocated_items_.end()) {
                delete *it;
                allocated_items_.erase(it);
            }
        }

        size_t get_usage() const {
            return allocated_items_.size() * sizeof(T);
        }
    };

} // namespace ar_slam