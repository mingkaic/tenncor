#include "eigen/eigen.hpp"

#include "dbg/peval/plugin_eval.hpp"

#ifndef DBG_STATS_HPP
#define DBG_STATS_HPP

namespace stats
{

template <typename T>
void inspect_helper (T* data, teq::Shape shape, const std::string& label)
{
	if (nullptr == data)
	{
		global::errorf("cannot inspect null data of shape %s",
			shape.to_string().c_str());
	}
	size_t n = shape.n_elems();
	T min = *std::min_element(data, data + n);
	T max = *std::max_element(data, data + n);
	// using logs is slow af (tolerable) (todo: upgrade to formatter whenever c++2a is supported)
	global::infof("(%s) => min: %s, max: %s",
		label.c_str(),
		fmts::to_string(min).c_str(),
		fmts::to_string(max).c_str());
}

#define _INSPECTOR_SELECT(T)\
inspect_helper((T*) func->device().data(), func->shape(), label);

struct Inspector final : public dbg::iPlugin
{
	void process (
		const teq::TensSetT& targets,
		const teq::TensSetT& visited) override
	{
		for (auto vis : visited)
		{
			if (auto func = dynamic_cast<teq::iFunctor*>(vis))
			{
				if (estd::has(insps_, func))
				{
					std::string label = insps_.at(func);
					auto dtype = (egen::_GENERATED_DTYPE)
						func->get_meta().type_code();
					TYPE_LOOKUP(_INSPECTOR_SELECT, dtype)
				}
			}
		}
	}

	std::unordered_map<teq::iFunctor*,std::string> insps_;
};

#undef _INSPECTOR_SELECT

}

#endif // DBG_STATS_HPP
