#include "teq/iopfunc.hpp"

#include "teq/mock/data.hpp"
#include "teq/mock/functor.hpp"

#ifndef TEQ_MOCK_OPFUNC_HPP
#define TEQ_MOCK_OPFUNC_HPP

struct MockOpfunc : public teq::iOperableFunc
{
	MockOpfunc (teq::TensptrsT tens,
		std::vector<double> data,
		teq::Opcode opcode = teq::Opcode{}) :
		func_(tens, opcode),
		data_(tens.front()->shape(), data) {}

	MockOpfunc (const MockOpfunc& other) :
		updated_(other.updated_), func_(other.func_), data_(other.data_) {}

	virtual ~MockOpfunc (void) = default;

	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(*this);
	}

	teq::Shape shape (void) const override
	{
		return func_.shape();
	}

	std::string to_string (void) const override
	{
		return func_.to_string();
	}

	teq::Opcode get_opcode (void) const override
	{
		return func_.get_opcode();
	}

	teq::TensptrsT get_children (void) const override
	{
		return func_.get_children();
	}

	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		return func_.get_attr(attr_name);
	}

	std::vector<std::string> ls_attrs (void) const override
	{
		return func_.ls_attrs();
	}

	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override
	{
		func_.add_attr(attr_key, std::move(attr_val));
	}

	void rm_attr (std::string attr_key) override
	{
		func_.rm_attr(attr_key);
	}

	void update_child (teq::TensptrT arg, size_t index) override
	{
		func_.update_child(arg, index);
	}

	void update (void) override
	{
		updated_ = true;
	}

	void* data (void) override
	{
		return data_.data();
	}

	const void* data (void) const override
	{
		return data_.data();
	}

	size_t type_code (void) const override
	{
		return data_.type_code();
	}

	std::string type_label (void) const override
	{
		return data_.type_label();
	}

	size_t nbytes (void) const override
	{
		return data_.nbytes();
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockOpfunc(*this);
	}

	bool updated_ = false;

	MockFunctor func_;

	MockData data_;
};

#endif // TEQ_MOCK_OPFUNC_HPP
