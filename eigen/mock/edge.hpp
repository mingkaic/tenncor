#include "eigen/edge.hpp"

template <typename T>
struct MockEdge final : public eigen::iEigenEdge<T>
{
	MockEdge (teq::TensptrT tensor,
		std::vector<T> data,
		teq::Shape outshape,
		std::vector<double> coorder = {}) :
		tensor_(tensor), data_(data), shape_(outshape),
		coorder_(coorder)
	{
		if (tensor_ == nullptr)
		{
			logs::fatal("cannot map a null tensor");
		}
	}

	/// Implementation of iEdge
	teq::Shape shape (void) const override
	{
		return tensor_->shape();
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		return tensor_;
	}

	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		if ("coorder" == attr_name)
		{
			return &coorder_;
		}
	}

	std::vector<std::string> ls_attrs (void) const override
	{
		return {"coorder"};
	}

	T* data (void) const override
	{
		return const_cast<T*>(data_.data());
	}

private:
	teq::TensptrT tensor_;

	std::vector<T> data_;

	teq::Shape shape_;

	marsh::NumArray<double> coorder_;
};
