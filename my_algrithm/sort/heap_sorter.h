//
// Created by xiangxx on 2022/6/26.
//

#ifndef CPPRACTICE_HEAP_SORTER_H
#define CPPRACTICE_HEAP_SORTER_H

#include "vector"

using namespace std;

class HeapSorter
{
public:
	template<typename T>
	static void HeapSort(vector<T>& array);

	template<typename T>
	static void HeapAdjust(vector<T>& array, int index, int last_index);

	template<typename T>
	static void SwapAt(vector<T>& array, int i, int j);
};

template<typename T>
void HeapSorter::HeapSort(vector<T>& array)
{
	int last_index = array.size() - 1;
	int last_node_with_child = (last_index - 1) / 2;
	for (int i = last_node_with_child; i >= 0; i--)
	{
		HeapAdjust(array, i, last_index);
	}

	while (last_index > 0)
	{
		SwapAt(array, 0, last_index);
		last_index--;
		HeapAdjust(array, 0, last_index);
	}
}

template<typename T>
void HeapSorter::HeapAdjust(vector<T>& array, int index, int last_index)
{
	int cur = index;
	int largest = cur;
	int left = index * 2 + 1;
	int right = left + 1;

	if (left <= last_index && array[largest] < array[left])
	{
		largest = left;
	}

	if (right <= last_index && array[largest] < array[right])
	{
		largest = right;
	}

	if (largest != cur)
	{
		SwapAt(array, cur, largest);
		HeapAdjust(array, largest, last_index);
	}
}

template<typename T>
void HeapSorter::SwapAt(vector<T>& array, int i, int j)
{
	auto temp = array[i];
	array[i] = array[j];
	array[j] = temp;
}

#endif //CPPRACTICE_HEAP_SORTER_H
