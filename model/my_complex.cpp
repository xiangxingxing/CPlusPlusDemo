//
// Created by xiangxx on 2022/6/11.
//

#include "my_complex.h"

//Complex& Complex::operator += (const Complex& other){
//	return _doapl(this, other);
//}

Complex::Complex(double re, double im) : re_(re), im_(im)
{

}

//output operator
ostream& operator<<(ostream& os, const Complex& c)
{
	return os << '(' << Real(c) << ',' << Imag(c) << ')';
}

inline Complex& _doapl(Complex* ths, const Complex& other)
{
	ths->re_ += other.re_;
	ths->im_ += other.im_;
	return *ths;
}
