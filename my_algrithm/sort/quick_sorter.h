//
// Created by xiangxx on 2022/4/5.
//

#ifndef CPPRACTICE_QUICK_SORTER_H
#define CPPRACTICE_QUICK_SORTER_H

#include "vector"

using namespace std;

class QuickSorter
{
public:
	template<typename T>
	static void QuickSortLomuto(vector<T>& array, int low, int high);

	template<typename T>
	static void QuickSortHoare(vector<T>& array, int low, int high);

	template<typename T>
	static void QuickSortBook(vector<T>& array, int low, int high);

private:
	template<typename T>
	static int QuickSortLomutoInternal(vector<T>& array, int low, int high);

	template<typename T>
	static int QuickSortHoareInternal(vector<T>& array, int low, int high);

	template<typename T>
	static int QuickSortBookInternal(vector<T>& array, int low, int high);

	template<typename T>
	static void SwapAt(vector<T>& array, int i, int j);
};

template<typename T>
void QuickSorter::QuickSortLomuto(vector<T>& array, int low, int high)
{
	if (low < high)
	{
		int pivot_index = QuickSortLomutoInternal(array, low, high);
		QuickSortLomuto(array, low, pivot_index - 1);
		QuickSortLomuto(array, pivot_index + 1,  high);
	}
}

template<typename T>
int QuickSorter::QuickSortLomutoInternal(vector<T>& array, int low, int high)
{
	//use last as pivot
	T pivot = array[high];
	int i = low;
	for (int j = low; j < high; ++j)
	{
		if (array[j] <= pivot)
		{
			QuickSorter::SwapAt(array, i, j);
			i++;
		}
	}

	QuickSorter::SwapAt(array, i, high);
	return i;
}


template<typename T>
void QuickSorter::QuickSortHoare(vector<T>& array, int low, int high)
{
	if (low < high)
	{
		int pivot_index = QuickSortHoareInternal(array, low, high);
		QuickSortHoare(array, low, pivot_index);
		QuickSortHoare(array, pivot_index + 1,  high);
	}
}

//fewer swap operation
template<typename T>
int QuickSorter::QuickSortHoareInternal(vector<T>& array, int low, int high)
{
	//use first as pivot
	T pivot = array[low];
	int i = low - 1;
	int j = high + 1;

	while (true)
	{
		do
		{
			j--;
		}
		while (array[j] > pivot);

		do
		{
			i++;
		}
		while (array[i] < pivot);

		if (i < j)
		{
			SwapAt(array, i, j);
		}
		else
		{
			return j;
		}
	}
}


template<typename T>
void QuickSorter::QuickSortBook(vector<T>& array, int low, int high)
{
	if (low < high)
	{
		int pivot_index = QuickSortBookInternal(array, low, high);
		QuickSortBook(array, low, pivot_index - 1);
		QuickSortBook(array, pivot_index + 1, high);
	}
}


template<typename T>
int QuickSorter::QuickSortBookInternal(vector<T>& array, int low, int high)
{
	int index = (low + high) / 2;
	SwapAt(array, low, index);
	T pivot = array[low];

	while (low < high)
	{
		while (low < high && array[high] >= pivot)
		{
			high--;
		}

		array[low] = array[high];

		while (low < high && array[low] <= pivot)
		{
			low++;
		}

		array[high] = array[low];

	}

	array[low] = pivot;

	return low;
}

template<typename T>
void QuickSorter::SwapAt(vector<T>& array, int i, int j)
{
	auto temp = array[i];
	array[i] = array[j];
	array[j] = temp;
}

#endif //CPPRACTICE_QUICK_SORTER_H
