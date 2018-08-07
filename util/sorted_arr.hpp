#include <array>
#include <unordered_set>

#include "util/error.hpp"

#ifndef SORTED_ARR_HPP
#define SORTED_ARR_HPP

template <typename T, size_t N>
struct SortedArr
{
    SortedArr (std::initializer_list<T> list)
    {
        size_t n = list.size();
        if (n != N)
        {
            handle_error("creating sorted array from mismatched list",
                ErrArg<size_t>("list_size", n),
                ErrArg<size_t>("range", N));
        }
        std::memcpy(arr_, list.begin(), N * sizeof(T));
        std::sort(arr_, arr_ + N);
    }

    SortedArr (std::array<T,N> arr)
    {
        std::memcpy(arr_, &arr[0], N * sizeof(T));
        std::sort(arr_, arr_ + N);
    }

    T& operator [] (size_t i)
    {
        if (i >= N)
        {
            handle_error("accessing index exceeds range",
                ErrArg<size_t>("index", i),
                ErrArg<size_t>("range", N));
        }
        return arr_[i];
    }

    const T& operator [] (size_t i) const
    {
        if (i >= N)
        {
            handle_error("accessing index exceeds range",
                ErrArg<size_t>("index", i),
                ErrArg<size_t>("range", N));
        }
        return arr_[i];
    }

private:
    T arr_[N];
};

#endif /* SORTED_ARR_HPP */
