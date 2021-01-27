
#ifndef EIGEN_METADATA_HPP
#define EIGEN_METADATA_HPP

#include "internal/teq/teq.hpp"
#include "internal/eigen/generated/dtype.hpp"

namespace eigen
{

template <typename T>
struct EMetadata : public teq::iMetadata
{
	EMetadata (size_t version = 0) : version_(version) {}

	/// Implementation of iMetadata
	size_t type_code (void) const override
	{
		return egen::get_type<T>();
	}

	/// Implementation of iMetadata
	std::string type_label (void) const override
	{
		return egen::name_type(egen::get_type<T>());
	}

	/// Implementation of iMetadata
	size_t type_size (void) const override
	{
		return sizeof(T);
	}

	/// Implementation of iMetadata
	size_t state_version (void) const override
	{
		return version_;
	}

	size_t version_ = 0;
};

struct EMetadata2 : public teq::iMetadata
{
	EMetadata2 (egen::_GENERATED_DTYPE dtype, size_t version) :
		dtype_(dtype), version_(version) {}

	/// Implementation of iMetadata
	size_t type_code (void) const override
	{
		return dtype_;
	}

	/// Implementation of iMetadata
	std::string type_label (void) const override
	{
		return egen::name_type(dtype_);
	}

	/// Implementation of iMetadata
	size_t type_size (void) const override
	{
		return egen::type_size(dtype_);
	}

	/// Implementation of iMetadata
	size_t state_version (void) const override
	{
		return version_;
	}

	egen::_GENERATED_DTYPE dtype_;

	size_t version_;
};

}

#endif // EIGEN_METADATA_HPP
