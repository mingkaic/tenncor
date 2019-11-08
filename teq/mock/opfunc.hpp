#include "teq/iopfunc.hpp"
#include "teq/mock/edge.hpp"

#ifndef TEQ_MOCK_OPFUNC_HPP
#define TEQ_MOCK_OPFUNC_HPP

struct MockOpfunc final : public teq::iOperableFunc
{
	MockOpfunc (teq::TensptrT a,
		teq::Opcode opcode = teq::Opcode{},
		teq::CvrtptrT coord = nullptr) :
		args_({MockEdge(a, teq::identity, coord)}),
		opcode_(opcode) {}

	MockOpfunc (teq::TensptrT a, teq::TensptrT b,
		teq::Opcode opcode = teq::Opcode{}) :
		args_({MockEdge(a), MockEdge(b)}),
		shape_(args_[0].shape()), opcode_(opcode) {}

	MockOpfunc (MockEdgesT args,
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
	teq::CEdgesT get_children (void) const override
	{
		return teq::CEdgesT(args_.begin(), args_.end());
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override {}

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

	MockEdgesT args_;

	teq::Shape shape_;

	teq::Opcode opcode_;
};

#endif // TEQ_MOCK_OPFUNC_HPP
