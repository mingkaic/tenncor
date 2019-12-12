#include "eigen/edge.hpp"

template <typename T>
struct MockEdge final : public eigen::iEigenEdge<T>
{
	MockEdge (teq::TensptrT tensor,
		std::vector<T> data) :
		tensor_(tensor), data_(data)
	{
		if (tensor_ == nullptr)
		{
			logs::fatal("cannot map a null tensor");
		}
	}

	teq::Shape shape (void) const override
	{
		return tensor_->shape();
	}

	T* data (void) const override
	{
		return const_cast<T*>(data_.data());
	}

private:
	teq::TensptrT tensor_;

	std::vector<T> data_;
};
