#include <iostream>
/*struct Str2
{
    Str2* operator -> ()
    {
        return this;
    }
    int bla = 10;
};
struct Str
{
    Str(int* p)
        :ptr(p)
        {}
    int& operator * ()
    {
        return *ptr;
    }
    Str2 operator -> ()
    {

        return Str2{};
    }
    int val = 5;
private:
    int* ptr;
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
*/
struct Base
{
};
struct Derive: public Base
{
};
int main()
{  
    Derive d;
    Base& ref = d;
    Base* ptr = &d;
}