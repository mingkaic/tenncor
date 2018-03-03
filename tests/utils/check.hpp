#include <cstdarg>
#include <sstream>

#include "tensor/tensorshape.hpp"

#ifndef TTEST_CHECK_HPP
#define TTEST_CHECK_HPP

namespace testutils
{

template <typename T>
void print (std::vector<T> raw, std::ostream& os = std::cout)
{
	if (raw.empty())
	{
		os << "empty";
	}
	else
	{
		for (T r : raw)
		{
			os << r << " ";
		}
		os << "\n";
	}
}

std::string sprintf (const char* fmt...);

bool tensorshape_equal (
	const nnet::tensorshape& ts1,
	const nnet::tensorshape& ts2);

bool tensorshape_equal (
	const nnet::tensorshape& ts1,
	std::vector<size_t>& ts2);

}

#endif
