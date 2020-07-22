#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "eigen/eigen.hpp"

#ifndef DISTRIB_IREFERENCE_HPP
#define DISTRIB_IREFERENCE_HPP

namespace distr
{

const std::string refname = "DISTRIB_REFERENCE";

struct iDistRef : public teq::iFunctor
{
	virtual ~iDistRef (void) = default;

	/// Implementation of iTensor
	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(*this);
	}

	/// Implementation of iAttributed
	std::vector<std::string> ls_attrs (void) const override
	{
		return {};
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (const std::string& attr_key) const override
	{
		return nullptr;
	}

	/// Implementation of iAttributed
	marsh::iObject* get_attr (const std::string& attr_key) override
	{
		return nullptr;
	}

	/// Implementation of iAttributed
	void add_attr (const std::string& attr_key,
		marsh::ObjptrT&& attr_val) override {}

	/// Implementation of iAttributed
	void rm_attr (const std::string& attr_key) override {}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return teq::Opcode{refname, 0};
	}

	/// Implementation of iFunctor
	teq::TensptrsT get_args (void) const override
	{
		return {};
	}

	/// Implementation of iFunctor
	teq::TensptrsT get_dependencies (void) const override
	{
		return {};
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override {}

	virtual void update_data (const double* data, size_t version) = 0;

	/// Return string id of cluster owner
	virtual const std::string& cluster_id (void) const = 0;

	virtual const std::string& node_id (void) const = 0;
};

using DRefptrT = std::shared_ptr<iDistRef>;

using DRefptrSetT = std::unordered_set<DRefptrT>;

#define CACHE_UPDATE(cache_type)\
{ std::vector<cache_type> tmp(data, data + nelems);\
std::memcpy(cache_.data(), &tmp[0], sizeof(cache_type) * nelems); }

struct DistRef final : public iDistRef
{
	DistRef (egen::_GENERATED_DTYPE dtype, teq::Shape shape,
		const std::string& cluster_id, const std::string& self_id) :
		cache_(shape.n_elems() * egen::type_size(dtype)), shape_(shape),
		cluster_id_(cluster_id), self_(self_id), meta_(dtype, 0) {}

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
		if (version > meta_.version_)
		{
			size_t nelems = shape_.n_elems();
			TYPE_LOOKUP(CACHE_UPDATE, meta_.dtype_)
			meta_.version_ = version;
		}
	}

	/// Implementation of iDistRef
	const std::string& cluster_id (void) const override
	{
		return cluster_id_;
	}

	const std::string& node_id (void) const override
	{
		return self_;
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

DRefptrSetT reachable_refs (teq::TensptrT root);

}

#endif // DISTRIB_IREFERENCE_HPP
