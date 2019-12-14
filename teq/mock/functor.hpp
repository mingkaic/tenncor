#include "marsh/objs.hpp"

#include "teq/ifunctor.hpp"

#ifndef TEQ_MOCK_FUNCTOR_HPP
#define TEQ_MOCK_FUNCTOR_HPP

struct MockFunctor : public teq::iFunctor
{
	MockFunctor (teq::TensptrsT tens, teq::Opcode opcode = teq::Opcode{}) :
		opcode_(opcode), shape_(tens.front()->shape()), tens_(tens) {}

	MockFunctor (const MockFunctor& other) :
		opcode_(other.opcode_),
		shape_(other.shape_),
		tens_(other.tens_)
	{
		std::unique_ptr<marsh::Maps> oattr(other.attrs_.clone());
		attrs_ = std::move(*oattr);
	}

	virtual ~MockFunctor (void) = default;

	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(*this);
	}

	teq::Shape shape (void) const override
	{
		return shape_;
	}

	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	teq::Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	teq::TensptrsT get_children (void) const override
	{
		return tens_;
	}

	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		return attrs_.get_attr(attr_name);
	}

	std::vector<std::string> ls_attrs (void) const override
	{
		return attrs_.ls_attrs();
	}

	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override
	{
		attrs_.add_attr(attr_key, std::move(attr_val));
	}

	void rm_attr (std::string attr_key) override
	{
		attrs_.rm_attr(attr_key);
	}

	void update_child (teq::TensptrT arg, size_t index) override
	{
		tens_[index] = arg;
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockFunctor(*this);
	}

	teq::Opcode opcode_;

	teq::Shape shape_;

	teq::TensptrsT tens_;

	marsh::Maps attrs_;
};

#endif // TEQ_MOCK_FUNCTOR_HPP
