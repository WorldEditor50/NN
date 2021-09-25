#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <iostream>
#include <random>
#include <memory>
#include "allocator.hpp"


template <typename T, template<typename> class TAllocator = Allocator>
class Vector
{
public:
    Vector():ptr(nullptr), size_(0){}
    explicit Vector(size_t N):ptr(allocator.allocate(N)), size_(N){}
    explicit Vector(size_t N, T x):ptr(allocator.allocate(N)), size_(N)
    {
        for (int i = 0; i < size_; i++) {
            ptr[i] = x;
        }
    }
    ~Vector()
    {
        if (ptr != nullptr) {
            allocator.deallocate(size_, ptr);
        }
    }
    Vector(const Vector &r):ptr(allocator.allocate(r.size_)), size_(r.size_)
    {
        for (int i = 0; i < size_; i++) {
            ptr[i] = r.ptr[i];
        }
    }
    Vector(Vector &&r):ptr(r.ptr),size_(r.size_)
    {
        r.ptr = nullptr;
        r.size_ = 0;
    }
    Vector& operator = (const Vector &r)
    {
        if (this == &r) {
            return *this;
        }
        if (size_ != r.size_) {
            allocator.deallocate(size_, ptr);
            ptr = allocator.allocate(r.size_);
            size_ = r.size_;
        }
        for (int i = 0; i < size_; i++) {
            ptr[i] = r.ptr[i];
        }
        return *this;
    }
    Vector& operator = (Vector &&r)
    {
        if (this == &r) {
            return *this;
        }
        ptr = r.ptr;
        size_ = r.size_;
        r.ptr = nullptr;
        r.size_ = 0;
        return *this;
    }

    Vector operator + (const Vector &r)
    {
        if (size_ != r.size_) {
            return *this;
        }
        Vector y(r.size_);
        for (int i = 0; i < size_; i++) {
            y.ptr[i] = ptr[i] + r.ptr[i];
        }
        return y;
    }

    Vector operator - (const Vector &r)
    {
        if (size_ != r.size_) {
            return *this;
        }
        Vector y(r.size_);
        for (int i = 0; i < size_; i++) {
            y.ptr[i] = ptr[i] - r.ptr[i];
        }
        return y;
    }

    Vector operator * (const Vector &r)
    {
        if (size_ != r.size_) {
            return *this;
        }
        Vector y(r.size_);
        for (int i = 0; i < size_; i++) {
            y.ptr[i] = ptr[i] * r.ptr[i];
        }
        return y;
    }

    Vector operator / (const Vector &r)
    {
        if (size_ != r.size_) {
            return *this;
        }
        Vector y(r.size_);
        for (int i = 0; i < size_; i++) {
            y.ptr[i] = ptr[i] / r.ptr[i];
        }
        return y;
    }

    Vector& operator += (const Vector &r)
    {
        if (size_ != r.size_) {
            return *this;
        }
        for (int i = 0; i < size_; i++) {
            ptr[i] += r.ptr[i];
        }
        return *this;
    }

    Vector& operator -= (const Vector &r)
    {
        if (size_ != r.size_) {
            return *this;
        }
        for (int i = 0; i < size_; i++) {
            ptr[i] -= r.ptr[i];
        }
        return *this;
    }

    Vector& operator *= (const Vector &r)
    {
        if (size_ != r.size_) {
            return *this;
        }
        for (int i = 0; i < size_; i++) {
            ptr[i] *= r.ptr[i];
        }
        return *this;
    }

    Vector& operator /= (const Vector &r)
    {
        if (size_ != r.size_) {
            return *this;
        }
        for (int i = 0; i < size_; i++) {
            ptr[i] /= r.ptr[i];
        }
        return *this;
    }

    Vector operator + (T x)
    {
        Vector y(size_);
        for (int i = 0; i < size_; i++) {
            y.ptr[i] = ptr[i] + x;
        }
        return y;
    }

    Vector operator - (T x)
    {
        Vector y(size_);
        for (int i = 0; i < size_; i++) {
            y.ptr[i] = ptr[i] - x;
        }
        return y;
    }

    Vector operator * (T x)
    {
        Vector y(size_);
        for (int i = 0; i < size_; i++) {
            y.ptr[i] = ptr[i] * x;
        }
        return y;
    }

    Vector operator / (T x)
    {
        Vector y(size_);
        for (int i = 0; i < size_; i++) {
            y.ptr[i] = ptr[i] / x;
        }
        return y;
    }

    Vector& operator += (T x)
    {
        for (int i = 0; i < size_; i++) {
            ptr[i] += x;
        }
        return *this;
    }

    Vector& operator -= (T x)
    {
        for (int i = 0; i < size_; i++) {
            ptr[i] -= x;
        }
        return *this;
    }

    Vector& operator *= (T x)
    {
        for (int i = 0; i < size_; i++) {
            ptr[i] *= x;
        }
        return *this;
    }

    Vector& operator /= (T x)
    {
        for (int i = 0; i < size_; i++) {
            ptr[i] /= x;
        }
        return *this;
    }

    void show()
    {
        for (int i = 0; i < size_; i++) {
            std::cout<<ptr[i]<<" ";
        }
        std::cout<<std::endl;
        return;
    }
    void rand(int N)
    {
        std::default_random_engine e;
        std::uniform_int_distribution<int> r(0, N);
        for (int i = 0; i < size_; i++) {
            ptr[i] = r(e);
        }
        return;
    }
    void rand(float N)
    {
        std::default_random_engine e;
        std::uniform_real_distribution<float> r(0, N);
        for (int i = 0; i < size_; i++) {
            ptr[i] = r(e);
        }
        return;
    }
    void rand(double N)
    {
        std::default_random_engine e;
        std::uniform_real_distribution<double> r(0, N);
        for (int i = 0; i < size_; i++) {
            ptr[i] = r(e);
        }
        return;
    }
    inline size_t size() const {return size_;}
    inline T& operator[](size_t i) {return ptr[i];}
    inline T& at(size_t i) {return ptr[i];}
private:
    static TAllocator<T> allocator;
    T *ptr;
    size_t size_;
};

template <typename T, template<typename> class TAllocator>
TAllocator<T> Vector<T, TAllocator>::allocator;

using Vectori = Vector<int>;
using Vectorf = Vector<float>;
using Vectord = Vector<double>;

template<typename T>
T dot(const Vector<T> &x1, const Vector<T> &x2)
{
    T s = 0;
    for (int i = 0; i < x1.size_; i++) {
        s += x1.ptr[i] * x2.ptr[i];
    }
    return s;
}
#endif // VECTOR_HPP
