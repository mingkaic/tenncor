#include "opt/rule_src.hpp"

#ifndef EAD_CONVERSION_HPP
#define EAD_CONVERSION_HPP

namespace ead
{

namespace opt
{

struct RuleConversion
{
	RuleptrT source_;

	//
};

template <typename T>
void apply_rules (ead::Nodes<T>& nodes,
    std::vector<RuleConversion>& conversions)
{
    std::unordered_map<RepptrT,size_t> graphsize;
    std::unordered_map<std::string,std::string> imprints;
    {
        Representer<T> representer;
        for (auto& node : nodes)
        {
            auto tens = node->get_tensor();
            tens->accept(representer);
        }
        for (auto& reppair : representer.reps_)
        {
            auto rep = reppair.second;
        }
        imprints = representer.imprints_;
    }

    size_t min_height = std::numeric_limits<size_t>::max();
    for (auto& rule_convs : conversions)
    {
        size_t height = rule_convs.source_->get_height();
        if (min_height > height)
        {
            min_height = height;
        }
    }

    //
}

}

}

#endif // EAD_CONVERSION_HPP
