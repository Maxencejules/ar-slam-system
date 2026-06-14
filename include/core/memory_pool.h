#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace ar_slam {

/**
 * @brief Fixed-capacity object pool with O(1) allocation and deallocation.
 *
 * Backing storage is a single contiguous, over-aligned slab sized at
 * construction time from a byte budget. Free slots are threaded into an
 * intrusive singly linked free-list, so both allocate() and deallocate() are
 * constant time and never touch the heap after construction. The capacity is a
 * hard limit: once the slab is exhausted, allocate()/create() return nullptr
 * instead of growing, which makes the pool suitable for latency-sensitive and
 * memory-constrained (e.g. embedded) frame pipelines.
 *
 * The pool manages raw storage only: allocate()/deallocate() do not construct
 * or destruct objects. Use create()/destroy() for the common case where you
 * want construction and destruction handled for you.
 *
 * The type is non-copyable (it owns a unique slab) but movable.
 *
 * @tparam T Object type stored in the pool.
 */
template <typename T>
class MemoryPool {
public:
    /**
     * @brief Construct a pool whose slab is sized to fit within @p max_bytes.
     * @param max_bytes Storage budget in bytes (default 256 MiB). The realised
     *                  capacity is floor(max_bytes / slot_size) objects.
     */
    explicit MemoryPool(std::size_t max_bytes = 256ull * 1024 * 1024)
        : capacity_(max_bytes / kSlotSize) {
        if (capacity_ == 0) {
            capacity_ = 1;  // Always provide room for at least one object.
        }
        slots_ = static_cast<Slot*>(
            ::operator new(capacity_ * sizeof(Slot), std::align_val_t{kAlign}));

        // Thread every slot onto the free-list, front to back.
        for (std::size_t i = 0; i + 1 < capacity_; ++i) {
            slots_[i].next = &slots_[i + 1];
        }
        slots_[capacity_ - 1].next = nullptr;
        free_head_ = &slots_[0];
    }

    ~MemoryPool() {
        ::operator delete(slots_, std::align_val_t{kAlign});
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    MemoryPool(MemoryPool&& other) noexcept
        : slots_(other.slots_),
          free_head_(other.free_head_),
          capacity_(other.capacity_),
          used_(other.used_) {
        other.slots_ = nullptr;
        other.free_head_ = nullptr;
        other.capacity_ = 0;
        other.used_ = 0;
    }

    MemoryPool& operator=(MemoryPool&& other) noexcept {
        if (this != &other) {
            ::operator delete(slots_, std::align_val_t{kAlign});
            slots_ = other.slots_;
            free_head_ = other.free_head_;
            capacity_ = other.capacity_;
            used_ = other.used_;
            other.slots_ = nullptr;
            other.free_head_ = nullptr;
            other.capacity_ = 0;
            other.used_ = 0;
        }
        return *this;
    }

    /**
     * @brief Reserve one slot of raw, uninitialised storage.
     * @return Pointer to storage for a T, or nullptr if the pool is exhausted.
     */
    T* allocate() noexcept {
        if (free_head_ == nullptr) {
            return nullptr;
        }
        Slot* slot = free_head_;
        free_head_ = slot->next;
        ++used_;
        return reinterpret_cast<T*>(slot);
    }

    /**
     * @brief Return a slot obtained from allocate() to the free-list.
     * @param ptr Pointer previously returned by allocate()/create(); nullptr is
     *            ignored. Does not call the destructor.
     */
    void deallocate(T* ptr) noexcept {
        if (ptr == nullptr) {
            return;
        }
        Slot* slot = reinterpret_cast<Slot*>(ptr);
        slot->next = free_head_;
        free_head_ = slot;
        --used_;
    }

    /**
     * @brief Allocate a slot and construct a T in place.
     * @return Pointer to the constructed object, or nullptr if the pool is full.
     *         If the constructor throws, the slot is returned to the pool and
     *         the exception propagates.
     */
    template <typename... Args>
    T* create(Args&&... args) {
        T* storage = allocate();
        if (storage == nullptr) {
            return nullptr;
        }
        try {
            return ::new (storage) T(std::forward<Args>(args)...);
        } catch (...) {
            deallocate(storage);
            throw;
        }
    }

    /**
     * @brief Destroy an object created with create() and reclaim its slot.
     */
    void destroy(T* ptr) noexcept {
        if (ptr == nullptr) {
            return;
        }
        ptr->~T();
        deallocate(ptr);
    }

    /// Maximum number of objects the pool can hold.
    std::size_t capacity() const noexcept { return capacity_; }

    /// Number of slots currently handed out.
    std::size_t used() const noexcept { return used_; }

    /// Number of slots still available.
    std::size_t available() const noexcept { return capacity_ - used_; }

    /// True when no further allocations can succeed.
    bool full() const noexcept { return used_ == capacity_; }

    /// Bytes currently in use by live objects.
    std::size_t get_usage() const noexcept { return used_ * sizeof(T); }

    /// Total bytes reserved by the backing slab.
    std::size_t capacity_bytes() const noexcept { return capacity_ * sizeof(Slot); }

    /// True if @p ptr points into this pool's slab.
    bool owns(const T* ptr) const noexcept {
        const auto* p = reinterpret_cast<const Slot*>(ptr);
        return p >= slots_ && p < slots_ + capacity_;
    }

private:
    // A slot is either live object storage or, when free, a free-list node.
    // The union sizes/aligns to satisfy both roles.
    union Slot {
        Slot* next;
        alignas(T) unsigned char storage[sizeof(T)];
    };

    static constexpr std::size_t kSlotSize = sizeof(Slot);
    static constexpr std::size_t kAlign =
        alignof(T) > alignof(Slot*) ? alignof(T) : alignof(Slot*);

    Slot* slots_ = nullptr;
    Slot* free_head_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t used_ = 0;
};

}  // namespace ar_slam
