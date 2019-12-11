#include "marsh/objs.hpp"

#include "teq/iedge.hpp"

#ifndef TEQ_MOCK_EDGE_HPP
#define TEQ_MOCK_EDGE_HPP

struct MockEdge final : public teq::iEdge
{
	MockEdge (teq::TensptrT tensor,
		std::vector<double> shape = {},
		std::vector<double> coorder = {},
		std::vector<double> junk = {}) :
		tensor_(tensor),
		shape_(shape), coorder_(coorder), junk_coords_(junk)
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

	teq::TensptrT get_tensor (void) const override
	{
		return tensor_;
	}

	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		if ("shape" == attr_name && shape_.contents_.size() > 0)
		{
			return &shape_;
		}
		else if ("coorder" == attr_name && coorder_.contents_.size() > 0)
		{
			return &coorder_;
		}
		else if ("junkcoorder" == attr_name && junk_coords_.contents_.size() > 0)
		{
			return &junk_coords_;
		}
		return nullptr;
	}

	std::vector<std::string> ls_attrs (void) const override
	{
		return {"shape", "coorder", "junkcoorder"};
	}

	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override {}

	void rm_attr (std::string attr_key) override {}

private:
	teq::TensptrT tensor_;

	marsh::NumArray<double> shape_;

	marsh::NumArray<double> coorder_;

	marsh::NumArray<double> junk_coords_;
};

using MockEdgesT = std::vector<MockEdge>;

#endif // TEQ_MOCK_EDGE_HPP
