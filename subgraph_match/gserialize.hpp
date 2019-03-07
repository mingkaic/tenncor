#include "ade/ade.hpp"

#include "subgraph_match/pattern.hpp"

#ifndef OPT_GSERIAL_HPP
#define OPT_GSERIAL_HPP

namespace opt
{

struct GraphSerializer final : ade::iTraveler
{
    GraphSerializer (TransformsT transforms) : transforms_(transforms) {}

	/// Implementation of iTraveler
    void visit (ade::iLeaf* leaf) override
	{
        if (serials_.end() == serials_.find(leaf))
        {
            serials_.emplace(leaf, DepthMatrixT{
                std::vector<std::string>{encode_leaf(leaf)}});
        }
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
        if (serials_.end() == serials_.find(func))
        {
            DepthMatrixT serials = {
                std::vector<std::string>{encode_func(func)}};
            auto children = func->get_children();
            for (auto child : children)
            {
                auto smart_tens = child.get_tensor();
                auto tens = smart_tens.get();
                smart_map_.emplace(tens, smart_tens);
                tens->accept(*this);
                auto& child_serials = serials_[tens];
                for (size_t i = 0, n = child_serials.size(); i < n; ++i)
                {
                    auto& child_serial = child_serials[i];
                    if (i + 1 >= tens.size())
                    {
                        serials.push_back(child_serial);
                    }
                    else
                    {
                        auto& serial = serials[i + 1];
                        serial.insert(serial.end(),
                            child_serial.begin(), child_serial.end());
                    }
                }
            }
            // simplify serial and save
            serials_.emplace(func,
                simplify_depthmatrix(transforms_, serials));
        }
	}

    TransformsT transforms_;

    std::unordered_map<ade::iTensor*,DepthMatrixT> serials_;

    std::unordered_map<ade::iTensor*,ade::TensptrT> smart_map_;
};

}

#endif // OPT_GSERIAL_HPP
