
#include "ade/shape.hpp"

#include "ead/generated/data.hpp"

#include "ead/eigen.hpp"

#ifndef EAD_TENSOR_HPP
#define EAD_TENSOR_HPP

namespace ead
{

template <typename T>
inline TensorT<T> make_tensor (const ade::Shape& shape)
{
	std::array<Eigen::Index,ade::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	TensorT<T> out(slist);
	out.setZero();
	return out;
}

template <typename T>
inline MatMapT<T> make_matmap (T* data, const ade::Shape& shape)
{
	if (nullptr == data)
	{
		logs::fatal("cannot get tensmap from nullptr");;
	}
	return MatMapT<T>(data, shape.at(1), shape.at(0));
}

template <typename T>
inline TensMapT<T> make_tensmap (T* data, const ade::Shape& shape)
{
	std::array<Eigen::Index,ade::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	if (nullptr == data)
	{
		logs::fatal("cannot get tensmap from nullptr");;
	}
	return TensMapT<T>(data, slist);
}

template <typename T>
ade::Shape get_shape (const TensorT<T>& tens)
{
	auto slist = tens.dimensions();
	return ade::Shape(std::vector<ade::DimT>(slist.begin(), slist.end()));
}

template <typename T>
ade::Shape get_shape (const TensMapT<T>& tens)
{
	auto slist = tens.dimensions();
	return ade::Shape(std::vector<ade::DimT>(slist.begin(), slist.end()));
}

DimensionsT shape_convert (ade::Shape shape);

}

#endif // EAD_TENSOR_HPP
