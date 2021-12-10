#include <iostream>

void f1()
{
    throw 1;

}
void f2()
{
    f1();
}
void f3()
{
    f2();
}

int main()
{
    try{
        f3();
    }
    catch(int)
    {
        std::cout << "123" << std::endl;
    }
    
}