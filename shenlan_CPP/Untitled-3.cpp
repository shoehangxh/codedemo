#include <iostream>
#include <type_traits>
#include <utility>

struct str
{
    const static int internel = 3;
};
int p = 5;

template <typename T>
void fun()
{
    std::cout << (T::internel* p) <<std::endl; //*有乘法和指针的歧义
}

int main()
{
    fun<str>();
    return 0;
}