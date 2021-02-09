#include "tenncor/pyutils/convert.hpp"

#ifdef PYTUTIL_CONVERT_HPP

namespace pyutils
{

teq::DimsT c2pshape (const teq::Shape& cshape)
{
	auto slist = teq::narrow_shape(cshape);
	std::reverse(slist.begin(), slist.end());
	return slist;
}

teq::Shape p2cshape (const py::list& pyshape)
{
	teq::DimsT slist;
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
	teq::DimsT slist(pslist, pslist + ndim);
	std::reverse(slist.begin(), slist.end());
	return teq::Shape(slist);
}

teq::Shape p2cshape2 (const py::list& pyshape)
{
	teq::DimsT slist;
	size_t n = pyshape.size();
	slist.reserve(n);
	for (size_t i = 0; i < n; ++i)
	{
		slist.push_back(pyshape[n - i - 1].cast<teq::DimT>());
	}
	return teq::Shape(slist);
}

teq::Shape p2cshape2 (const py::ssize_t* pslist, size_t ndim)
{
	teq::DimsT slist(pslist, pslist + ndim);
	std::reverse(slist.begin(), slist.end());
	return teq::Shape(slist);
}

}

#endif
