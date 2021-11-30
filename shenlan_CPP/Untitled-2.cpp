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
    Base ()
    {
        fun();
    }
    virtual void fun(int x = 3) {
        std::cout << "Base: " << x << std::endl;
    }   
};
struct Derive :Base
{
    Derive ()
        :Base()
    {
        fun();
    }
    void fun(int x = 4) override
    {
        std::cout << "Derive: " << x << std::endl;
    }
};
void proc(Base& b)
{
    b.fun();
}
int main()
{ 
   Derive d;
   //proc(d);
}