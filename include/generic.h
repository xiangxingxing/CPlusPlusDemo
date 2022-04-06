//
// Created by DEV on 2022/4/6.
//

#ifndef CPPRACTICE_GENERIC_H
#define CPPRACTICE_GENERIC_H

using namespace std;
#include <random>
#include <iostream>

#define Max(a, b) ( (a > b) ? a : b )
#define Min(a, b) ( (a < b) ? a : b )

#define RANDOM_INIT()	srand(time(NULL))
#define RANDOM(L, R)	(L + rand() % ((R) - (L) + 1)) // gen a random integer in [L, R]

#define RANDOM11(L, R) \
({ \
	std::random_device rd;\
	std::mt19937 mt(rd());\
	std::uniform_int_distribution<int> dist((L), (R)); \
	dist(mt);                        \
})

#endif //CPPRACTICE_GENERIC_H
