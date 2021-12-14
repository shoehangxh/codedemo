# include <iostream>
# include <string>
# include <cstring>
# include <vector>
# include <new>
# include <memory>
# include <map>
# include <unordered_set>
//# include <ranges>

using namespace std;
/*
std::shared_ptr<int> fun()
{
    std::shared_ptr<int>(new int (3));
    return res;
}

void fun(int* ptr)
{
    std::cout << "call deleter fun" <<endl;
    delete ptr;
}

int main()
{
    std::shared_ptr<int> x = fun();
    std::shared_ptr<int> x(new int(3), fun);
}

struct Str{
    int x;
    int y;
};
class str{
public:
    str(){
    }
    str(int input){
        x = input;
    }
private:
    int x;
};
*/
class Str{
    Str() :ptr(new int()) {}
    ~Str() {delete ptr;}
    Str(const Str& val) :ptr(new int()) //拷贝构造
    {
        *ptr = *(val.ptr);
    }
    Str& operator= （const Str& val）//拷贝赋值
    {
        *ptr = *(val.ptr);
        return *this;
    }
    Str(Str&& val) noexcept //移动构造（移动赋值与其不存在太大的性能差别）
    :ptr(val.ptr)
    {
        val.ptr = nullptr;
    }

    int& Data()
    {
        return *ptr
    }
private:
    int* ptr;
}
int main()
{
    Str.a;
    a.Data() = 3;
    Str b(a);//会出错，因为b.ptr所指向的内存与a指向的内存一致（自动生成的拷贝构造函数），
    //但销毁的时候先销毁b，所以内存会重复销毁,因此要自己加一个拷贝构造函数/拷贝赋值函数
    b = a; // 拷贝赋值
    std::cout << a.Data << std::endl; 
    /*Str m_str;
    m_str.x = 3;
    std::cout << m_str.x << std::endl; 
    int* ptr = new int(3);
    int* ptr1 = new int[5];
    int a = 4;
    std::cout << sizeof(ptr) <<endl; //8
    std::cout << sizeof(ptr1) <<endl; //8
    std::cout << sizeof(a) <<endl; //4

    allocator<int> al;
    int* ptr2 = al.allocate(3); //只返回内存，不会构造
    
    std::vector<int> a{1};
    std::vector<int> b{1, 0, 0};
    int v = -1;

    cout << (a < b) <<endl;
    //cout << v;
    
    std::map<int, bool> m{{3,true}, {4,false},{1,true}};
    for (auto p : m){
        cout << p.first << ' ' <<p.second <<endl; 
    }
    std::unordered_set<int> s{3, 1, 5, 4, 1};
    for (auto p : s){
        cout << p << endl;
    }

    int y = 10;
    int z = 3;
    auto x = [y+z] (int val) mutable{
        ++y;
        return val > y;
        };
    std::cout << x(5) <<std::endl;
    auto lam = []<typename T>(T val) {return val +1;};
    std::vector<int> x{1,2,3, 4, 5};
    auto it = std::ranges::find(x, 3);
    std::cout << *it << std::endl;*/
    return 0;

}













/*int main()
{
    
    int* y = new int(2);
    cout << *y <<endl;
    delete y;
    int* y = new(std::nothrow) int[5];
    delete[] y;
    char ch[sizeof(int)];
    int* y = new (ch) int(4);

    int* x = 0;
    delete x;

    int* ptr = new int[5];
    int* ptr2 = (ptr + 1)
    delete[] ptr2; //error

    std::shared_ptr<int> x (new int(3));





}*/