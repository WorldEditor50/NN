#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP
#include <map>
#include <vector>

template <typename T>
class Allocator
{
public:
    Allocator(){}
    T* allocate(size_t N)
    {
        T* ptr = nullptr;
        if (mem.find(N) == mem.end()) {
            ptr = new T[N];
        } else {
            if (mem[N].empty()) {
                ptr = new T[N];
            } else {
                ptr = mem[N].back();
                mem[N].pop_back();
            }
        }
        return ptr;
    }
    void deallocate(size_t N, T* &ptr)
    {
        if (N == 0 || ptr == nullptr) {
            return;
        }
        mem[N].push_back(ptr);
        ptr = nullptr;
        return;
    }
    ~Allocator()
    {
        for (auto it = mem.begin(); it != mem.end(); it++) {
            std::vector<T*> &block = it->second;
            for (auto* ptr : block) {
                delete [] ptr;
            }
            block.clear();
        }
        mem.clear();
    }
private:
    std::map<size_t, std::vector<T*> > mem;
};

#endif // ALLOCATOR_HPP
