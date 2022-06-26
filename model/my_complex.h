//
// Created by xiangxx on 2022/6/11.
//

#ifndef CPPRACTICE_MY_COMPLEX_H
#define CPPRACTICE_MY_COMPLEX_H

#include <iostream>
using namespace std;

class Complex{
private:
	double re_;
	double im_;

	friend Complex& _doapl(Complex* ths, const Complex& other);

public:
	explicit Complex(double re = 0, double im = 0);

	inline double Real() const{
		return re_;
	}

	inline void SetReal(double re){
		re_ = re;
	}

	inline double Imag() const{
		return im_;
	}

	inline void SetImag(double im){
		im_ = im;
	}

	/*
	 * inline void operator += (const Complex& other);
	 * 也可以返回void,但是就不能实现 C1 += C2 += C3 的连串形式
	 * */
	inline Complex& operator += (const Complex& other){
		return _doapl(this, other);
	}

	/*
	 * ‼️输出需要设计为全局函数‼️
	 * */
	friend ostream& operator<<(ostream& os, const Complex& c);
};

/*
 * 非成员函数[无this]
 * */

inline double Real(const Complex& complex)
{
	return complex.Real();
}

inline double Imag(const Complex& complex)
{
	return complex.Imag();
}

//不能返回reference

inline Complex operator +(const Complex& c1, const Complex& c2)
{
	return Complex(Real(c1) + Real(c2), Imag(c1) + Imag(c2));
}

inline Complex operator +(double x, const Complex& c2)
{
	return Complex(x + Real(c2), Imag(c2));
}

inline Complex operator +(const Complex& c1, double y)
{
	return Complex(Real(c1) + y, Imag(c1));
}

//正，也可以返回引用&
inline Complex operator + (const Complex& c)
{
	return c;
}
//负
inline Complex operator - (const Complex& c)
{
	return Complex(-(Real(c), -(Imag(c))));
}

inline bool operator ==(const Complex& x, const Complex& y)
{
	return Real(x) == Real(y) && Imag(x) == Imag(y);
}

inline bool operator ==(double x, const Complex& y)
{
	return Real(y) == x && Imag(y) == 0;
}

inline bool operator ==(const Complex& x, double y)
{
	return Real(x) == y && Imag(x) == 0;
}


inline bool operator !=(const Complex& x, const Complex& y)
{
	return Real(x) != Real(y) || Imag(x) != Imag(y);
}

inline bool operator !=(double x, const Complex& y)
{
	return Real(y) != x || Imag(y) != 0;
}

inline bool operator !=(const Complex& x, double y)
{
	return Real(x) != y || Imag(x) != 0;
}

//共轭复数
inline Complex conj(const Complex& x)
{
	return Complex(Real(x), -Imag(x));
}



#endif //CPPRACTICE_MY_COMPLEX_H
