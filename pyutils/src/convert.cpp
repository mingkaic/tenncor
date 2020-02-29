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

teq::Shape p2cshape (const py::list& pyshape)
{
	std::vector<teq::DimT> slist;
	size_t n = pyshape.size();
	slist.reserve(n);
	for (size_t i = 0; i < n; ++i)
	{
		slist.push_back(pyshape[n - i - 1].cast<teq::DimT>());
	}
	return teq::Shape(slist);
}

teq::Shape p2cshape (const py::ssize_t* pslist, size_t ndim)
{
	std::vector<teq::DimT> slist(pslist, pslist + ndim);
	std::reverse(slist.begin(), slist.end());
	return teq::Shape(slist);
}

}

#endif
