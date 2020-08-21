#include "internal/teq/teq.hpp"
#include "internal/eigen/generated/dtype.hpp"

#ifndef EIGEN_METADATA_HPP
#define EIGEN_METADATA_HPP

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

}

#endif // EIGEN_METADATA_HPP
