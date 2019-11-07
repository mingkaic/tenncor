#include "teq/iopfunc.hpp"

#ifndef TEQ_MOCK_OPFUNC_HPP
#define TEQ_MOCK_OPFUNC_HPP

struct MockOpfunc final : public teq::iOperableFunc
{
	MockOpfunc (teq::TensptrT a,
		teq::Opcode opcode = teq::Opcode{},
		teq::CoordptrT coord = teq::identity) :
		args_({teq::FuncArg(a, teq::identity, false, coord)}),
		opcode_(opcode) {}

	MockOpfunc (teq::TensptrT a, teq::TensptrT b,
		teq::Opcode opcode = teq::Opcode{}) :
		args_({teq::identity_map(a), teq::identity_map(b)}),
		shape_(args_[0].shape()), opcode_(opcode) {}

	MockOpfunc (teq::ArgsT args,
		teq::Opcode opcode = teq::Opcode{}) :
		args_(args),
		shape_(args_[0].shape()), opcode_(opcode) {}

	/// Implementation of iTensor
	const teq::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	/// Implementation of iFunctor
	teq::CstArgsT get_children (void) const override
	{
		return teq::CstArgsT(args_.begin(), args_.end());
	}

	/// Implementation of iFunctor
	void update_child (const teq::FuncArg& arg, size_t index) override {}

	/// Implementation of iOperableFunc
	void update (void) override
	{
		updated_ = true;
	}

	/// Implementation of iData
	void* data (void) override
	{
		return nullptr;
	}

	/// Implementation of iData
	const void* data (void) const override
	{
		return nullptr;
	}

	/// Implementation of iData
	size_t type_code (void) const override
	{
		return 0;
	}

	/// Implementation of iData
	std::string type_label (void) const override
	{
		return "";
	}

	/// Implementation of iData
	size_t nbytes (void) const override
	{
		return 0;
	}

	bool updated_ = false;

	teq::ArgsT args_;

	teq::Shape shape_;

	teq::Opcode opcode_;
};

#endif // TEQ_MOCK_OPFUNC_HPP
