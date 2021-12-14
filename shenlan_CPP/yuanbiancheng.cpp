#include <iostream>
#include <type_traits>

template <int x>
constexpr auto fun = (x % 2) + fun<x / 2>;

template <>
constexpr auto fun<0> = 0;

constexpr auto x = fun<99>;

int main()
{
    std::cout << x <<std::endl; //4
}

