#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "eigen/eigen.hpp"

#ifndef DISTRIB_IREFERENCE_HPP
#define DISTRIB_IREFERENCE_HPP

namespace distrib
{

struct iDistRef : public teq::iLeaf
{
	virtual ~iDistRef (void) = default;

	/// Implementation of iTensor
	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(*this);
	}

	/// Implementation of iLeaf
	teq::Usage get_usage (void) const override
	{
		return teq::PLACEHOLDER;
	}

	virtual void update_data (const double* data, size_t version) = 0;

	/// Return string id of cluster owner
	virtual const std::string& cluster_id (void) const = 0;
};

using DRefptrT = std::shared_ptr<iDistRef>;

using DRefsT = std::vector<DRefptrT>;

#define CACHE_UPDATE(cache_type)\
{ std::vector<cache_type> tmp(data, data + nelems);\
std::memcpy(cache_.data(), &tmp[0], sizeof(cache_type) * nelems); }

struct DistRef final : public iDistRef
{
	DistRef (egen::_GENERATED_DTYPE dtype, teq::Shape shape,
		const std::string& cluster_id, const std::string& self_id) :
		cache_(shape.n_elems() * egen::type_size(dtype)), shape_(shape),
		cluster_id_(cluster_id), self_(self_id), meta_(dtype, 1) {}

	/// Implementation of iTensor
	teq::iDeviceRef& device (void) override
	{
		return cache_;
	}

	/// Implementation of iTensor
	const teq::iDeviceRef& device (void) const override
	{
		return cache_;
	}

	/// Implementation of iTensor
	const teq::iMetadata& get_meta (void) const override
	{
		return meta_;
	}

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return cluster_id_ + "/" + self_;
	}

	void update_data (const double* data, size_t version) override
	{
		size_t nelems = shape_.n_elems();
		TYPE_LOOKUP(CACHE_UPDATE, meta_.dtype_)
		meta_.version_ = version;
	}

	/// Implementation of iDistRef
	const std::string& cluster_id (void) const override
	{
		return cluster_id_;
	}

private:
	struct DataCache final : teq::iDeviceRef
	{
		DataCache (size_t nbytes) : data_(nbytes , 0) {}

		/// Implementation of iDeviceRef
		void* data (void) override
		{
			return &data_[0];
		}

		/// Implementation of iDeviceRef
		const void* data (void) const override
		{
			return data_.data();
		}

		std::string data_;
	};

	struct ExplicitMetadata : public teq::iMetadata
	{
		ExplicitMetadata (egen::_GENERATED_DTYPE dtype, size_t version = 0) :
			version_(version), dtype_(dtype) {}

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

		size_t version_ = 0;

		egen::_GENERATED_DTYPE dtype_;
	};

	teq::iTensor* clone_impl (void) const override
	{
		return new DistRef(egen::_GENERATED_DTYPE(meta_.dtype_),
			shape_, cluster_id_, self_);
	}

	DataCache cache_;

	teq::Shape shape_;

	std::string cluster_id_;

	std::string self_;

	/// Variable metadata
	ExplicitMetadata meta_;
};

}

#endif // DISTRIB_IREFERENCE_HPP
