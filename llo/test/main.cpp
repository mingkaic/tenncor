#include <iostream>

#include "llo/api.hpp"

int main (int argc, char** argv)
{
	auto src = Source<double>::get(ade::Shape({4, 3}),
		std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
	auto dest = sin(src);

	GenericData out = evaluate(DOUBLE, dest.get());
	double* optr = (double*) out.data_.get();
	std::cout << out.shape_.to_string() << std::endl;
	for (size_t i = 0, n = out.shape_.n_elems(); i < n; ++i)
	{
		std::cout << optr[i] << ",";
	}
	std::cout << std::endl;

	auto gsrc = dest->gradient(src);

	GenericData gout = evaluate(DOUBLE, gsrc.get());
	double* goptr = (double*) gout.data_.get();
	std::cout << gout.shape_.to_string() << std::endl;
	for (size_t i = 0, n = gout.shape_.n_elems(); i < n; ++i)
	{
		std::cout << goptr[i] << ",";
	}
	std::cout << std::endl;

	return 0;
}
