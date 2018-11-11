#include <memory>

#include "ade/tensor.hpp"

#include "llo/dtype.hpp"

#ifndef LLO_DATA_HPP
#define LLO_DATA_HPP

namespace llo
{

/// GenericData for holding data when passing up the tensor graph
struct GenericData final : public ade::iData
{
	GenericData (void) = default;

	GenericData (ade::Shape shape, DTYPE dtype, std::string label);

	void take_astype (DTYPE outtype, const ade::iData& other);

	char* get (void) override
	{
		return data_.get();
	}

	const char* get (void) const override
	{
		return data_.get();
	}

	size_t type_code (void) const override
	{
		return dtype_;
	}

	/// Smartpointer to a block of untyped data
	std::shared_ptr<char> data_;

	/// Shape of data_
	ade::Shape shape_;

	/// Type encoding of data_
	DTYPE dtype_;

	std::string label_;
};

/// GenericRef for holding data
/// Ref uses raw pointer instead of shared, so it's memory unsafe
struct GenericRef
{
	GenericRef (char* data, ade::Shape shape, DTYPE dtype) :
		data_(data), shape_(shape), dtype_(dtype) {}

	GenericRef (GenericData& generic) :
		data_(generic.data_.get()),
		shape_(generic.shape_), dtype_(generic.dtype_) {}

	/// Raw pointer to a block of untyped data
	char* data_;

	/// Shape of data_
	ade::Shape shape_;

	/// Data type of data_
	DTYPE dtype_;
};

template <typename T>
struct DataNode final : public ade::Tensor
{
	static DataNode<T>* get (std::vector<T> data, ade::Shape shape)
	{
		return new DataNode<T>(data, shape, "unlabelled");
	}

	static DataNode<T>* get (std::vector<T> data, ade::Shape shape,
		std::string label)
	{
		return new DataNode<T>(data, shape, label);
	}

	DataNode (const DataNode&) = default;
	DataNode (DataNode&&) = default;
	DataNode& operator = (const DataNode&) = default;
	DataNode& operator = (DataNode&&) = default;

	/// Assign vectorized data to source
	DataNode& operator = (std::vector<T> data)
	{
		GenericRef ref((char*) &data[0], shape(), get_type<T>());
		return operator = (ref);
	}

	DataNode& operator = (GenericRef data)
	{
		if (false == data.shape_.compatible_after(shape(), 0))
		{
			err::fatalf("cannot assign data of incompatible shaped %s to "
				"internal data of shape %s", data.shape_.to_string().c_str(),
				shape().to_string().c_str());
		}
		std::memcpy(data_.data_.get(), data.data_,
			sizeof(T) * data.shape_.n_elems());
	}

	ade::iData& data (void) override
	{
		return data_;
	}

private:
	DataNode (std::vector<T> data, ade::Shape shape, std::string label) :
		ade::Tensor(shape), data_(shape, get_type<T>(), label)
	{
		std::memcpy(data_.data_.get(),
			&data[0], sizeof(T) * shape.n_elems());
	}

	GenericData data_;
};

template <typename T>
using VariableT = std::shared_ptr<llo::DataNode<T>>;

template <typename T>
DataNode<T>* data (T scalar, ade::Shape shape, std::string label)
{
	return llo::DataNode<T>::get(std::vector<T>(shape.n_elems(),scalar),
		shape, label);
}

}

#endif // LLO_DATA_HPP
