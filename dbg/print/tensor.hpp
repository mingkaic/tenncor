///
/// tensor.hpp
/// dbg
///
/// Purpose:
/// Draw tensor data in python-array format
///

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#ifndef DBG_TENSOR_HPP
#define DBG_TENSOR_HPP

/// Draw data as a multi-dimension array (similar to python) according to shape
template <typename T>
struct PrettyTensor final
{
	/// Datalimit represents the maximum number of elements to print for each
	/// dimension corresponding to datalimit's index
	/// For example: given datalimit={5, 4, 3, 2},
	/// print will only display at most 5 elements of the first dimension,
	/// at most 4 of the second, etc. beyond the first four dimensions,
	/// printing will display as many elements as there are
	PrettyTensor (std::vector<uint16_t> datalimit) : datalimit_(datalimit) {}

	/// Given the output stream, flattened tensor data, and the shape,
	/// wrap square brackets around each dimension in the tensor and stream to out
	/// For example: given arr={1, 2, 3, 4}, shape={2, 2},
	/// print will stream [[1, 2], [3, 4]] to out
	void print (std::ostream& out, T* arr, std::vector<uint8_t> shape)
	{
		print_helper(out, shape, arr, shape.size() - 1);
	}

	/// Number of elements to show for each dimension
	std::vector<uint16_t> datalimit_;

private:
	void print_helper (std::ostream& out, const std::vector<uint8_t>& shape,
		T* arr, uint8_t rank)
	{
		out << "[";
		uint16_t n = shape[rank];
		// apply limit only if limit is available
		if (rank < datalimit_.size())
		{
			n = std::min(n, datalimit_[rank]);
		}
		if (rank == 0)
		{
			out << arr[0];
			for (uint16_t i = 1; i < n; ++i)
			{
				out << "," << arr[i];
			}
		}
		else
		{
			auto it = shape.begin();
			size_t before = std::accumulate(it, it + rank, (size_t) 1,
				std::multiplies<size_t>());
			print_helper(out, shape, arr, rank - 1);
			for (uint16_t i = 1; i < n; ++i)
			{
				out << ",";
				print_helper(out, shape, arr + i * before, rank - 1);
			}
		}
		if (n < shape[rank])
		{
			out << "..";
		}
		out << "]";
	}
};

#endif // DBG_TENSOR_HPP
