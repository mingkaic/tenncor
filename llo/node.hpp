///
/// node.hpp
/// llo
///
/// Purpose:
/// Extend ade::iTensor with proxies that carry and evaluate tensor data
///

#include "llo/eval.hpp"

#ifndef LLO_NODE_HPP
#define LLO_NODE_HPP

namespace llo
{

/// Leaf evaluable holding tensor data
template <typename T>
struct Source final : public iSource
{
	/// Return a Source of input shape and containing input data
	static DataNode get (ade::Shape shape, std::vector<T> data)
	{
		if (shape.n_elems() != data.size())
		{
			ade::fatalf("data size %d does not match shape %s",
				data.size(), shape.to_string().c_str());
		}
		auto tens = std::shared_ptr<ade::Tensor>(ade::Tensor::get(shape));
		auto src = std::shared_ptr<iSource>(new Source(tens, data));
		return DataNode(EvalCtx(tens.get(), src),
			std::static_pointer_cast<ade::iTensor>(tens));
	}

	/// Return a source instance with a scalar as tensor data
	static DataNode get_scalar (T value)
	{
		return Source<T>::get(ade::Shape(), {value});
	}

	static DataNode copy (const llo::Source<T>& other)
	{
		auto rawsrc = new Source<T>(other);
		auto src = std::shared_ptr<iSource>(rawsrc);
		std::shared_ptr<ade::Tensor> tens = rawsrc->inner();
		return DataNode(EvalCtx(tens.get(), src),
			std::static_pointer_cast<ade::iTensor>(tens));
	}

	/// Implementation of iSource
	GenericData data (DTYPE dtype) const override
	{
		DTYPE curtype = get_type<T>();
		GenericData out(tensor_->shape(), curtype);
		std::memcpy(out.data_.get(), &data_[0], sizeof(T) * data_.size());
		if (curtype != dtype)
		{
			return out.convert_to(dtype);
		}
		return out;
	}

	/// Implementation of iSource
	DTYPE native_type (void) const override
	{
		return get_type<T>();
	}

	/// Implementation of iSource
	void reassign (const GenericRef& data) override
	{
		const ade::Shape& internal_shape = tensor_->shape();
		if (false == data.shape_.compatible_after(internal_shape, 0))
		{
			ade::fatalf("cannot assign data of incompatible shaped %s to "
				"internal data of shape %s", data.shape_.to_string().c_str(),
				internal_shape.to_string().c_str());
		}
		std::memcpy(&data_[0], data.data_, sizeof(T) * data.shape_.n_elems());
	}

	/// Implementation of iSource
	const std::shared_ptr<ade::Tensor>& inner (void) const override
	{
		return tensor_;
	}

private:
	Source (std::shared_ptr<ade::Tensor>& tensor, std::vector<T>& data) :
		tensor_(tensor), data_(data) {}

	Source (const Source<T>& other) :
		tensor_(other.tensor_), data_(other.data_) {}

	/// Tensor node of subgraph
	std::shared_ptr<ade::Tensor> tensor_;

	/// Tensor data
	std::vector<T> data_;
};

/// DataNode of a leaf that can be assignable by data vectors
template <typename T>
struct PlaceHolder : public DataNode
{
	PlaceHolder (ade::Shape shape) :
		DataNode(Source<T>::get(shape, std::vector<T>(shape.n_elems()))) {}

	PlaceHolder (const PlaceHolder&) = default;
	PlaceHolder (PlaceHolder&&) = default;
	PlaceHolder& operator = (const PlaceHolder&) = default;
	PlaceHolder& operator = (PlaceHolder&&) = default;

	/// Assign vectorized data to source
	PlaceHolder& operator = (std::vector<T>& data)
	{
		auto key = static_cast<ade::Tensor*>(tensor_.get());
		auto src = ctx_.srcs_[key];
		GenericRef gdata((char*) &data[0], tensor_->shape(), get_type<T>());
		src->reassign(gdata);
		return *this;
	}
};

template <typename T>
DataNode shaped_scalar (T scalar, ade::Shape shape)
{
	DataNode snode = Source<double>::get_scalar(scalar);
	return DataNode(snode.ctx_, ade::Functor::get(ade::COPY, {
		{ade::extend(0, std::vector<ade::DimT>(shape.begin(), shape.end())),
		snode.tensor_}}));
}

}

#endif // LLO_NODE_HPP
