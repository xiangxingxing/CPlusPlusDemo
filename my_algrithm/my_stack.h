//
// Created by xiangxx on 2022/6/4.
//

#ifndef CPPRACTICE_MY_STACK_H
#define CPPRACTICE_MY_STACK_H

#include <cstdint>
#include <exception>

using namespace std;

template<typename T>
class Stack
{
private:
	class StackEmptyException : public exception
	{
		public:
			virtual const char* what() const throw()
			{
				return "stack is empty";
			}
	} excp_empty;;

	class StackIndexOutOfBoundException : public exception
	{
		public:
			virtual const char* what() const throw()
			{
				return "Index out of bound.";
			}
	} excp_ioob;

	uint32_t capacity_;		// the total capacity
	uint32_t size_;			// current stack size
	T * elements_;		// the elements

public:
	explicit Stack(uint32_t capacity) :
		capacity_(capacity),
		size_(0),
		elements_(new T[capacity])
	{
	}

	~Stack()
	{
		delete[] elements_;
	}

	inline bool IsEmpty() const
	{
		return size_ == 0;
	}

	inline void Pop()
	{
		if (size_ != 0)
		{
			size_--;
		}
	}

	inline const T& Top() const
	{
		if (size_ == 0)
			throw excp_empty;

		return elements_[size_ - 1];
	}

	inline bool Push(const T& value)
	{
		if (size_ == capacity_)
		{
			return false;
		}
		else
		{
			elements_[size_++] = value;
			return true;
		}
	}

	inline uint32_t Count() const { return size_; }

	/**
	* return value by index from top
	*/
	inline const T& operator[](uint32_t index) const
	{
		if (index >= capacity_)
			throw excp_ioob;

		return elements_[size_ - 1 - index];
	}
};

#endif //CPPRACTICE_MY_STACK_H
