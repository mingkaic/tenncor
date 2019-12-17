#include "pyutils/convert.hpp"

#ifdef PYTUTIL_CONVERT_HPP

namespace pyutils
{

std::vector<teq::DimT> c2pshape (const teq::Shape& cshape)
{
	auto it = cshape.begin();
	auto et = cshape.end();
	while (it != et && *(et-1) == 1)
	{
		--et;
	}
	std::vector<teq::DimT> fwd(it, et);
	return std::vector<teq::DimT>(fwd.rbegin(), fwd.rend());
}

teq::Shape p2cshape (std::vector<py::ssize_t>& pyshape)
{
	return teq::Shape(std::vector<teq::DimT>(
		pyshape.rbegin(), pyshape.rend()));
}

}

#endif
