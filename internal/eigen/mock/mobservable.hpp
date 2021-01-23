
#ifndef EIGEN_MOCK_MOBSERVABLE_HPP
#define EIGEN_MOCK_MOBSERVABLE_HPP

#include "internal/eigen/mock/observable.hpp"

struct MockMObservable final : public eigen::Observable
{
	MockMObservable (void) : Observable(teq::TensptrsT{}) {}
	MockMObservable (const teq::TensptrsT& args) : Observable(args) {}
	MockMObservable (const teq::TensptrsT& args, marsh::Maps&& attrs) : Observable(args, std::move(attrs)) {}

	MockMObservable (const MockMObservable& other) : Observable(other) {}

	MockMObservable (MockMObservable&& other) : Observable(std::move(other)) {}

	MockMObservable& operator = (const MockMObservable& other)
	{
		if (this != &other)
		{
			Observable::operator = (other);
		}
		return *this;
	}

	MockMObservable& operator = (MockMObservable&& other)
	{
		if (this != &other)
		{
			Observable::operator = (std::move(other));
		}
		return *this;
	}

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
		return func_.has_data();
	}

	void uninitialize (void) override
	{
		func_.uninitialize();
	}

	bool initialize (void) override
	{
		return func_.initialize();
	}

	void must_initialize (void) override
	{
		func_.must_initialize();
	}

	void cache_init (void) override
	{
		func_.cache_init();
	}

	bool prop_version (size_t max_version) override
	{
		return func_.prop_version(max_version);
	}

	teq::iTensor* clone_impl (void) const override
	{
		return func_.clone_impl();
	}

	MockObservable func_;
};

#endif // EIGEN_MOCK_MOBSERVABLE_HPP
