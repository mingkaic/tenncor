#include "teq/iedge.hpp"

#ifndef TEQ_MOCK_EDGE_HPP
#define TEQ_MOCK_EDGE_HPP

struct MockEdge final : public teq::iEdge
{
	MockEdge (teq::TensptrT tensor,
        teq::ShaperT shaper = nullptr, teq::CvrtptrT coorder = nullptr) :
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
		if (nullptr == shaper_)
		{
			return argshape();
		}
		return shaper_->convert(argshape());
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
		if (nullptr != shaper_)
		{
			auto arr = std::make_unique<marsh::NumArray<teq::CDimT>>();
			auto& contents = arr->contents_;
			shaper_->access(
				[&](const teq::MatrixT& args)
				{
					for (teq::RankT i = 0; i < teq::mat_dim; ++i)
					{
						for (teq::RankT j = 0; j < teq::mat_dim; ++j)
						{
							contents.push_back(args[i][j]);
						}
					}
				});
			out.contents_.emplace("shape", std::move(arr));
		}
		if (nullptr != coorder_)
		{
			auto arr = std::make_unique<marsh::NumArray<teq::CDimT>>();
			auto& contents = arr->contents_;
			coorder_->access(
				[&](const teq::MatrixT& args)
				{
					for (teq::RankT i = 0; i < teq::mat_dim; ++i)
					{
						for (teq::RankT j = 0; j < teq::mat_dim; ++j)
						{
							contents.push_back(args[i][j]);
						}
					}
				});
			out.contents_.emplace("coord", std::move(arr));
		}
	}

private:
	teq::TensptrT tensor_;

	teq::ShaperT shaper_;

	teq::CvrtptrT coorder_;
};

using MockEdgesT = std::vector<MockEdge>;

#endif // TEQ_MOCK_EDGE_HPP
