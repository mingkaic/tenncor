#include "teq/iopfunc.hpp"
#include "teq/mock/edge.hpp"

#ifndef TEQ_MOCK_OPFUNC_HPP
#define TEQ_MOCK_OPFUNC_HPP

struct MockOpfunc final : public teq::iOperableFunc
{
	// todo: make constructor similar to MockFunctor
	MockOpfunc (teq::TensptrT a,
		teq::Opcode opcode = teq::Opcode{},
		std::vector<double> coord = {}) :
		MockOpfunc(teq::TensptrsT{a}, opcode, coord) {}

	MockOpfunc (teq::TensptrsT args,
		teq::Opcode opcode = teq::Opcode{},
		std::vector<double> coord = {}) :
		opcode_(opcode),
		shape_(args[0]->shape())
	{
		args_.reserve(args.size());
		std::transform(args.begin(), args.end(), std::back_inserter(args_),
			[](teq::TensptrT tens) { return MockEdge(tens); });
		if (coord.size() > 0)
		{
			coord_ = std::make_unique<marsh::NumArray<double>>();
			coord_->contents_ = coord;
		}
	}

	MockOpfunc (const MockOpfunc& other) :
		updated_(other.updated_),
		opcode_(other.opcode_),
		shape_(other.shape_),
		args_(other.args_),
		coord_(other.coord_->clone()) {}

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

	marsh::iObject* get_attr (std::string attr_name) const override
	{
		if (attr_name == "coorder")
		{
			return coord_.get();
		}
		return nullptr;
	}

	std::vector<std::string> ls_attrs (void) const override
	{
		return {"coorder"};
	}

	void update_child (teq::TensptrT arg, size_t index) override
	{
		args_[index] = MockEdge(arg);
	}

	void update (void) override
	{
		updated_ = true;
	}

	void* data (void) override
	{
		return nullptr;
	}

	const void* data (void) const override
	{
		return nullptr;
	}

	size_t type_code (void) const override
	{
		return 0;
	}

	std::string type_label (void) const override
	{
		return "";
	}

	size_t nbytes (void) const override
	{
		return 0;
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockOpfunc(*this);
	}

	bool updated_ = false;

	teq::Opcode opcode_;

	teq::Shape shape_;

	MockEdgesT args_;

	std::unique_ptr<marsh::NumArray<double>> coord_;
};

#endif // TEQ_MOCK_OPFUNC_HPP
