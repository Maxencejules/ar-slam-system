// Unit tests for the fixed-capacity object pool.
// Verifies capacity derivation, O(1) reuse from a real slab, enforced
// exhaustion, correct construction/destruction, and move semantics.

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

    std::size_t budget_for(std::size_t objects) {
        // Slot size is at least sizeof(void*); size the budget to that.
        std::size_t slot = sizeof(Tracked) > sizeof(void*) ? sizeof(Tracked) : sizeof(void*);
        return slot * objects;
    }

    void test_capacity_and_exhaustion() {
        ar_slam::MemoryPool<Tracked> pool(budget_for(4));
        CHECK(pool.capacity() == 4);
        CHECK(pool.used() == 0);
        CHECK(pool.available() == 4);
        CHECK(!pool.full());

        std::vector<Tracked*> objs;
        for (int i = 0; i < 4; ++i) {
            Tracked* t = pool.create(i * 10);
            CHECK(t != nullptr);
            CHECK(pool.owns(t));
            CHECK(t->value == i * 10);
            objs.push_back(t);
        }
        CHECK(pool.full());
        CHECK(Tracked::live == 4);

        // Exhaustion returns nullptr rather than growing the heap.
        CHECK(pool.create(99) == nullptr);
        CHECK(pool.allocate() == nullptr);

        // get_usage reflects only live objects.
        CHECK(pool.get_usage() == 4 * sizeof(Tracked));

        for (Tracked* t : objs) {
            pool.destroy(t);
        }
        CHECK(Tracked::live == 0);
        CHECK(pool.used() == 0);
    }

    void test_reuse() {
        ar_slam::MemoryPool<Tracked> pool(budget_for(3));
        Tracked* a = pool.create(1);
        Tracked* b = pool.create(2);
        Tracked* c = pool.create(3);
        CHECK(pool.full());

        pool.destroy(b);
        CHECK(pool.available() == 1);
        CHECK(Tracked::live == 2);

        Tracked* d = pool.create(42);
        CHECK(d != nullptr);
        CHECK(d->value == 42);
        CHECK(pool.full());

        pool.destroy(a);
        pool.destroy(c);
        pool.destroy(d);
        CHECK(Tracked::live == 0);
    }

    void test_move() {
        ar_slam::MemoryPool<Tracked> src(budget_for(2));
        Tracked* x = src.create(7);
        CHECK(x != nullptr);

        ar_slam::MemoryPool<Tracked> dst(std::move(src));
        CHECK(dst.owns(x));
        CHECK(x->value == 7);

        dst.destroy(x);
        CHECK(Tracked::live == 0);
    }

    void test_minimum_capacity() {
        // A tiny budget still yields room for at least one object.
        ar_slam::MemoryPool<Tracked> pool(1);
        CHECK(pool.capacity() >= 1);
        Tracked* t = pool.create(5);
        CHECK(t != nullptr);
        pool.destroy(t);
        CHECK(Tracked::live == 0);
    }

}  // namespace

int main() {
    test_capacity_and_exhaustion();
    test_reuse();
    test_move();
    test_minimum_capacity();
    return artest::report("test_memory_pool");
}
