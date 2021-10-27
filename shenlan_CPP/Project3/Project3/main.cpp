#include <iostream>
#include <vector>

void fun(std::vector<int>&& par) //右值引用
{

}

int main()
{
	/*std::vector<int> x;
	fun(std::move(x));  //x是将亡值，资源可以重新给别人
	const int y = 3;//行为不确定
	const int& ref = y;
	int z = 7;
	int* ptr = &z;
	float* ptr2 = reinterpret_cast<float*>(ptr);
	int& ref2 = const_cast<int&> (ref);
	ref2 = 4;
	std::cout << y << std::endl;
	std::cout << *ptr2 << std::endl; */

    const int y = 3;
	short x;
	x = { y };
	std::cout << x << std::endl;

}