#include "teq/ifunctor.hpp"
#include "teq/mock/edge.hpp"

#ifndef TEQ_MOCK_FUNCTOR_HPP
#define TEQ_MOCK_FUNCTOR_HPP

struct MockFunctor final : public teq::iFunctor
{
	MockFunctor (teq::TensptrsT tens,
		teq::Opcode opcode = teq::Opcode{}) :
		opcode_(opcode),
		shape_(tens[0]->shape()),
		args_(MockEdgesT(tens.begin(), tens.end())) {}

	MockFunctor (MockEdgesT args,
		teq::Opcode opcode = teq::Opcode{}) :
		opcode_(opcode),
		shape_(args[0].shape()),
		args_(args) {}

	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(this);
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

	teq::EdgeRefsT get_children (void) const override
	{
		return teq::EdgeRefsT(args_.begin(), args_.end());
	}

	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		return nullptr;
	}

	std::vector<std::string> ls_attrs (void) const override
	{
		return {};
	}

	void update_child (teq::TensptrT arg, size_t index) override
	{
		args_[index] = MockEdge(arg);
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockFunctor(*this);
	}

	teq::Opcode opcode_;

	teq::Shape shape_;

	MockEdgesT args_;
};

#endif // TEQ_MOCK_FUNCTOR_HPP
