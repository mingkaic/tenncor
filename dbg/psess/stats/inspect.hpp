#include "eigen/generated/dtype.hpp"

#include "dbg/psess/plugin_sess.hpp"

#ifndef DBG_STATS_HPP
#define DBG_STATS_HPP

namespace stats
{

template <typename T>
void inspect_helper (T* data, teq::Shape shape, std::string label)
{
	if (nullptr == data)
	{
		teq::errorf("cannot inspect null data of shape %s",
			shape.to_string().c_str());
	}
	size_t n = shape.n_elems();
	T min = *std::min_element(data, data + n);
	T max = *std::max_element(data, data + n);
	// using logs is slow af (tolerable) (todo: upgrade to formatter whenever c++2a is supported)
	teq::infof("(%s) => min: %s, max: %s",
		label.c_str(),
		fmts::to_string(min).c_str(),
		fmts::to_string(max).c_str());
}

#define INSPECTOR_SELECT(T)\
inspect_helper((T*) func->data(), func->shape(), label);

struct Inspector final : public dbg::iPlugin
{
	void process (teq::TensptrSetT& tracked, teq::FuncListT& targets) override
	{
		for (auto func : targets)
		{
			if (estd::has(insps_, func))
			{
				std::string label = insps_.at(func);
				auto dtype = (egen::_GENERATED_DTYPE) func->type_code();
				TYPE_LOOKUP(INSPECTOR_SELECT, dtype)
			}
		}
	}

	std::unordered_map<teq::iFunctor*,std::string> insps_;
};

#undef INSPECTOR_SELECT

}

#endif // DBG_STATS_HPP
