#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#include "core/memory_pool.h"
#include "test_util.h"

namespace {
    struct Tracked {
        static int live;
        int value;
        explicit Tracked(int v) : value(v) { ++live; }
        ~Tracked() { --live; }
    };
    int Tracked::live = 0;
    struct alignas(128) Aligned {
        int value = 0;
        unsigned char padding[124]{};
    };
    struct Throwing {
        explicit Throwing(bool fail) {
            if (fail)
                throw std::runtime_error("intentional constructor failure");
        }
    };
    std::size_t budget_for(std::size_t count) {
        return count * (sizeof(Tracked) > sizeof(void*) ? sizeof(Tracked) : sizeof(void*));
    }
    void test_budget_and_reuse() {
        for (std::size_t budget : {std::size_t{0}, std::size_t{1}, sizeof(void*) - 1}) {
            ar_slam::MemoryPool<Tracked> empty(budget);
            CHECK(empty.capacity() == 0);
            CHECK(empty.capacity_bytes() <= budget);
            CHECK(empty.allocate() == nullptr);
            CHECK(empty.create(1) == nullptr);
            CHECK(empty.full());
            CHECK(!empty.owns(nullptr));
        }
        ar_slam::MemoryPool<Tracked> pool(budget_for(4) + 1);
        CHECK(pool.capacity() == 4);
        CHECK(pool.capacity_bytes() <= budget_for(4) + 1);
        std::vector<Tracked*> objects;
        for (int i = 0; i < 4; ++i) {
            auto* object = pool.create(i);
            CHECK(object != nullptr);
            CHECK(pool.owns(object));
            CHECK(object->value == i);
            objects.push_back(object);
        }
        CHECK(pool.full());
        CHECK(pool.allocate() == nullptr);
        CHECK(pool.create(9) == nullptr);
        CHECK(pool.get_usage() == 4 * sizeof(Tracked));
        auto* reclaimed = objects[1];
        pool.destroy(reclaimed);
        auto* reused = pool.create(42);
        CHECK(reused == reclaimed);
        CHECK(reused->value == 42);
        objects[1] = reused;
        for (auto* object : objects)
            pool.destroy(object);
        CHECK(Tracked::live == 0);
        CHECK(pool.used() == 0);
        CHECK(pool.available() == 4);
        pool.deallocate(nullptr);
        Tracked foreign(0);
        CHECK(!pool.owns(&foreign));
        auto* raw = pool.allocate();
        auto* interior = reinterpret_cast<Tracked*>(reinterpret_cast<std::uintptr_t>(raw) + 1);
        CHECK(!pool.owns(interior));
        pool.deallocate(raw);
    }
    void test_alignment_and_constructor_rollback() {
        ar_slam::MemoryPool<Aligned> aligned(3 * sizeof(Aligned));
        CHECK(aligned.capacity() == 3);
        for (int i = 0; i < 3; ++i) {
            auto* value = aligned.create();
            CHECK(reinterpret_cast<std::uintptr_t>(value) % alignof(Aligned) == 0);
            aligned.destroy(value);
        }
        ar_slam::MemoryPool<Throwing> pool(sizeof(void*));
        CHECK(artest::throws<std::runtime_error>([&] { pool.create(true); }));
        CHECK(pool.used() == 0);
        auto* value = pool.create(false);
        CHECK(value != nullptr);
        CHECK(pool.full());
        pool.destroy(value);
        CHECK(pool.used() == 0);
    }
    void test_move() {
        ar_slam::MemoryPool<Tracked> source(budget_for(2));
        auto* value = source.create(7);
        ar_slam::MemoryPool<Tracked> target(std::move(source));
        CHECK(target.owns(value));
        CHECK(!source.owns(value));
        CHECK(source.capacity() == 0);
        CHECK(source.allocate() == nullptr);
        CHECK(value->value == 7);
        ar_slam::MemoryPool<Tracked> assigned(0);
        assigned = std::move(target);
        CHECK(assigned.owns(value));
        CHECK(target.capacity() == 0);
        assigned.destroy(value);
        CHECK(Tracked::live == 0);
        auto* alias = &assigned;
        assigned = std::move(*alias);
        CHECK(assigned.capacity() == 2);
    }
}  // namespace
int main() {
    test_budget_and_reuse();
    test_alignment_and_constructor_rollback();
    test_move();
    return artest::report("test_memory_pool");
}
