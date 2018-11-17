#include "age/test/grader_dep.hpp"
#include "age/test/codes_dep.hpp"

ade::Tensorptr cooler (size_t bardock)
{
	return new MockTensor(bardock, ade::Shape(
		std::vector<ade::DimT>{(ade::DimT) bardock}));
}
