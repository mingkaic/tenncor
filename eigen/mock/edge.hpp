#include "eigen/edge.hpp"

template <typename T>
struct MockEdge final : public eigen::iEigenEdge<T>
{
	MockEdge (teq::TensptrT tensor,
		std::vector<T> data,
		teq::Shape outshape,
		std::vector<double> coorder = {}) :
		tensor_(tensor), data_(data), shape_(outshape), coorder_(coorder)
	{
		if (tensor_ == nullptr)
		{
			logs::fatal("cannot map a null tensor");
		}
	}

	/// Implementation of iEdge
	teq::Shape shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iEdge
	teq::Shape argshape (void) const override
	{
		return tensor_->shape();
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		return tensor_;
	}

	/// Implementation of iEdge
	void get_attrs (marsh::Maps& out) const override
	{
		if (coorder_.size() > 0)
		{
			auto arr = std::make_unique<marsh::NumArray<double>>();
			auto& contents = arr->contents_;
			for (double c : coorder_)
			{
				contents.push_back(c);
			}
			out.contents_.emplace("coorder", std::move(arr));
		}
	}

	T* data (void) const override
	{
		return const_cast<T*>(data_.data());
	}

private:
	teq::TensptrT tensor_;

	std::vector<T> data_;

	teq::Shape shape_;

	std::vector<double> coorder_;
};
