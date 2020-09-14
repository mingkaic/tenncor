
#ifndef EIGEN_MOCK_OBSERVABLE_HPP
#define EIGEN_MOCK_OBSERVABLE_HPP

#include "internal/teq/mock/mock.hpp"

#include "internal/eigen/eigen.hpp"

struct MockObservable : public eigen::Observable
{
	MockObservable (void) : func_(
		teq::TensptrsT{std::make_shared<MockLeaf>(teq::Shape(), "A")},
		std::vector<double>{0}, teq::Opcode{}) {}

	MockObservable (teq::TensptrsT children,
		std::vector<double> data, teq::Opcode opcode) :
		func_(children, data, opcode) {}

	virtual ~MockObservable (void) = default;

	std::string to_string (void) const override
	{
		return func_.to_string();
	}

	teq::Opcode get_opcode (void) const override
	{
		return func_.get_opcode();
	}

	teq::TensptrsT get_args (void) const override
	{
		return func_.get_args();
	}

	void update_child (teq::TensptrT arg, size_t index) override
	{
		func_.update_child(arg, index);
	}

	teq::iDeviceRef& device (void) override
	{
		return func_.device();
	}

	const teq::iDeviceRef& device (void) const override
	{
		return func_.device();
	}

	const teq::iMetadata& get_meta (void) const override
	{
		return func_.get_meta();
	}

	teq::Shape shape (void) const override
	{
		return func_.shape();
	}

	bool has_data (void) const override
	{
		return data_;
	}

	void uninitialize (void) override
	{
		data_ = false;
	}

	bool initialize (void) override
	{
		if (succeed_initial_)
		{
			data_ = true;
		}
		return succeed_initial_;
	}

	void must_initialize (void) override
	{
		if (succeed_initial_)
		{
			data_ = true;
		}
	}

	bool prop_version (size_t max_version) override
	{
		if (max_version >= func_.meta_.version_)
		{
			return false;
		}
		++func_.meta_.version_;
		return succeed_prop_;
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockObservable(*this);
	}

	MockFunctor func_;

	bool data_ = false;

	bool succeed_initial_ = true;

	bool succeed_prop_ = true;
};

#endif // EIGEN_MOCK_OBSERVABLE_HPP
