
#ifndef EIGEN_MOCK_MUTABLE_LEAF_HPP
#define EIGEN_MOCK_MUTABLE_LEAF_HPP

#include "internal/teq/mock/mock.hpp"

#include "internal/eigen/eigen.hpp"

struct MockMutableLeaf : public eigen::iMutableLeaf
{
	MockMutableLeaf (void) :
		leaf_(), src_ref_((double*) leaf_.device().data(), teq::Shape()) {}

	MockMutableLeaf (teq::Shape shape, std::string label = "", bool cst = true) :
		leaf_(shape, label, cst),
		src_ref_((double*) leaf_.device().data(), shape) {}

	MockMutableLeaf (std::vector<double> data, teq::Shape shape,
		std::string label = "", bool cst = true) :
		leaf_(data, shape, label, cst),
		src_ref_((double*) leaf_.device().data(), shape) {}

	virtual ~MockMutableLeaf (void) = default;

	teq::Shape shape (void) const override
	{
		return leaf_.shape();
	}

	std::string to_string (void) const override
	{
		return leaf_.to_string();
	}

	teq::iDeviceRef& device (void) override
	{
		return src_ref_;
	}

	const teq::iDeviceRef& device (void) const override
	{
		return src_ref_;
	}

	const teq::iMetadata& get_meta (void) const override
	{
		return leaf_.get_meta();
	}

	teq::Usage get_usage (void) const override
	{
		return leaf_.get_usage();
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockMutableLeaf(*this);
	}

	void upversion (size_t version) override {}

	MockLeaf leaf_;

	eigen::SrcRef<double> src_ref_;
};

#endif // EIGEN_MOCK_MUTABLE_LEAF_HPP
