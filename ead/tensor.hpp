
#include "ade/shape.hpp"

#include "ead/generated/data.hpp"

#include "ead/eigen.hpp"

#ifndef EAD_TENSOR_HPP
#define EAD_TENSOR_HPP

namespace ead
{

template <typename T>
inline TensorT<T> get_tensor (const ade::Shape& shape)
{
	std::array<Eigen::Index,ade::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	TensorT<T> out(slist);
	out.setZero();
	return out;
}

template <typename T>
inline TensMapT<T> get_tensmap (T* data, const ade::Shape& shape)
{
	std::array<Eigen::Index,ade::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	if (nullptr == data)
	{
		logs::fatal("cannot get tensmap from nullptr");;
	}
	return Eigen::TensorMap<TensorT<T>>(data, slist);
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
