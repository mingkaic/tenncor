#include "ade/opfunc.hpp"
#include "ade/funcarg.hpp"
#include "ade/traveler.hpp"

#ifndef PRX_SUBGRAPH_HPP
#define PRX_SUBGRAPH_HPP

namespace prx
{

// Subgraph defines a border around all functors from root down to leaf,
// stopping at functors in treat_asleaf
struct Subgraph final : public ade::iOperableFunc
{
	static Subgraph* get (ade::Opcode graphcode, ade::TensptrT root,
        ade::TensT treat_asleaf = {})
	{
		return new Subgraph(graphcode, root, treat_asleaf);
	}

    /// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return root_->shape();
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return graphcode_.name_;
	}

	/// Implementation of iFunctor
	ade::Opcode get_opcode (void) const override
	{
		return graphcode_;
	}

	/// Implementation of iFunctor
	const ade::ArgsT& get_children (void) const override
	{
		return args_;
	}

	/// Implementation of iOperableFunc
	void update (void) override
	{
		for (ade::iOperableFunc*& op : ops_)
		{
			op->update();
		}
	}

	/// Implementation of iOperableFunc
	void* raw_data (void) override
	{
		return ops_.back()->raw_data();
	}

private:
    Subgraph (ade::Opcode graphcode, ade::TensptrT root,
        ade::TensT treat_asleaf) :
        graphcode_(graphcode), root_(root)
    {
		ade::GraphStat stat;
        for (ade::TensptrT& asleaf : treat_asleaf)
        {
            stat.graphsize_.emplace(asleaf.get(), ade::NumRange<size_t>(0, 0));
        }
        root->accept(stat);
		auto owners = ade::track_owners(root);
		for (auto& gsize_pair : stat.graphsize_)
		{
			if (0 == gsize_pair.second.upper_)
			{
				args_.push_back(ade::FuncArg(
					owners[gsize_pair.first].lock(),
					ade::identity));
			}
			else
			{
				auto opfunc = dynamic_cast<ade::iOperableFunc*>(
					gsize_pair.first);
				if (nullptr == opfunc)
				{
					logs::fatalf(
						"subgraph cannot track non-operable functor %s",
						gsize_pair.first->to_string());
				}
				ops_.push_back(opfunc);
			}
		}
		// ops_ go bottom up
		std::sort(ops_.begin(), ops_.end(),
			[&stat](ade::iOperableFunc* lhs, ade::iOperableFunc* rhs)
			{
				return stat.graphsize_[lhs].upper_ <
					stat.graphsize_[rhs].upper_;
			});
    }

    ade::Opcode graphcode_;

    ade::TensptrT root_;

    ade::ArgsT args_;

	std::vector<ade::iOperableFunc*> ops_;
};

}

#endif // PRX_SUBGRAPH_HPP
