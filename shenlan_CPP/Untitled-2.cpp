#include <iostream>
struct Str
{
    int val;


};
Str Add(Str x, Str y)
{
    Str z;
    z.val = x.val + y.val;
    return z;
}
auto operator + (Str x, Str y)
{
    Str z;
    z.val = x.val + y.val;
    return z;
}

int main()
{
    Str x;
    Str y;
    Str z = Add(x, y);
    Str a = x + y;
    std::cout << z.val <<std::endl;
    std::cout << a.val <<std::endl;
}