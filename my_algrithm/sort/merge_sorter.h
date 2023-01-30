//
// Created by xiangxx on 2023/1/28.
//

#ifndef CPPRACTICE_MERGE_SORTER_H
#define CPPRACTICE_MERGE_SORTER_H

#include "vector"

using namespace std;

class MergeSorter
{
public:
	template<typename T>
	static void Sort(vector<T>& array);

private:
	template<typename T>
	static void MergeSort(vector<T>& array, int start, int end, vector<T>& temp);

	template<typename T>
	static void InternalMerge(vector<T>& array, int start, int end, vector<T>& temp);
};

template<typename T>
void MergeSorter:: Sort(vector<T>& array)
{
	if(array.size() < 2){
		return;
	}

	vector<int> temp(array.size());
	MergeSort(array, 0, array.size() - 1, temp);
}

template<typename T>
void MergeSorter::MergeSort(vector<T>& array, int start, int end, vector<T>& temp)
{
	if (start >= end){
		return;
	}

	int mid_index = start + (end - start) / 2;
	MergeSort(array, start, mid_index, temp);
	MergeSort(array, mid_index + 1, end, temp);
	InternalMerge(array, start, end, temp);
}

template<typename T>
void MergeSorter::InternalMerge(vector<T>& array, int start, int end, vector<T>& temp)
{
	int mid_index = start + (end - start) / 2;
	int i = start;
	int j = mid_index + 1;
	int k = start;

	while(i <= mid_index && j <= end){
		temp[k++] = array[i] < array[j] ? array[i++] : array[j++];
	}

	while(i <= mid_index){
		temp[k++] = array[i++];
	}

	while(j <= end){
		temp[k++] = array[j++];
	}

	for (int t = start; t <= end; ++t)
	{
		array[t] = temp[t];
	}
}


#endif //CPPRACTICE_MERGE_SORTER_H
