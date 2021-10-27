#include <iostream>
#include <typeinfo>
#include <string>

extern int array[4];
int main()
{
	string a, b;
	cin >> a >> b;
	a[0] = 0;
	cout << a << b;
	system("pause");
}
