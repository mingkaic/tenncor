#include "teq/iedge.hpp"

#ifndef TEQ_MOCK_EDGE_HPP
#define TEQ_MOCK_EDGE_HPP

struct MockEdge final : public teq::iEdge
{
	MockEdge (teq::TensptrT tensor,
		std::vector<double> shaper = {},
		std::vector<double> coorder = {}) :
		tensor_(tensor), shaper_(shaper), coorder_(coorder)
	{
		if (tensor_ == nullptr)
		{
			logs::fatal("cannot map a null tensor");
		}
	}

	/// Implementation of iEdge
	teq::Shape shape (void) const override
	{
		return argshape();
	}

	/// Implementation of iEdge
	teq::Shape argshape (void) const override
	{
		return tensor_->shape();
	}

	/// Implementation of iEdge
	teq::TensptrT get_tensor (void) const override
	{
		return tensor_;
	}

	/// Implementation of iEdge
	void get_attrs (marsh::Maps& out) const override
	{
		if (shaper_.size() > 0)
		{
			auto arr = std::make_unique<marsh::NumArray<double>>();
			auto& contents = arr->contents_;
			for (double s : shaper_)
			{
				contents.push_back(s);
			}
			out.contents_.emplace("shaper", std::move(arr));
		}
		if (coorder_.size() > 0)
		{
			auto arr = std::make_unique<marsh::NumArray<double>>();
			auto& contents = arr->contents_;
			for (double c : coorder_)
			{
				contents.push_back(c);
			}
			out.contents_.emplace("coorder", std::move(arr));
		}
	}

private:
	teq::TensptrT tensor_;

	std::vector<double> shaper_;

	std::vector<double> coorder_;
};

using MockEdgesT = std::vector<MockEdge>;

#endif // TEQ_MOCK_EDGE_HPP
