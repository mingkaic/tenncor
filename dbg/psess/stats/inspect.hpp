#include "teq/ifunctor.hpp"

#include "eigen/generated/dtype.hpp"

#include "dbg/psess/plugin_sess.hpp"

#ifndef DBG_STATS_HPP
#define DBG_STATS_HPP

namespace stats
{

template <typename T>
void inspect_helper (T* data, teq::Shape shape)
{
	if (nullptr == data)
	{
		logs::errorf("cannot inspect null data of shape %s",
			shape.to_string().c_str());
	}
	size_t n = shape.n_elems();
	T min = *std::min_element(data, data + n);
	T max = *std::max_element(data, data + n);
	// using logs is slow af (tolerable) (todo: upgrade to formatter whenever c++2a is supported)
	logs::infof("min: %s, max: %s",
		fmts::to_string(min).c_str(),
		fmts::to_string(max).c_str());
}

#define INSPECTOR_SELECT(T)inspect_helper((T*) func->data(), func->shape());

struct Inspector final : public dbg::iPlugin
{
	void process (teq::TensptrSetT& tracked, teq::FuncListT& targets) override
	{
		for (auto func : targets)
		{
			if (estd::has(this->insps_, func))
			{
				auto dtype = (egen::_GENERATED_DTYPE) func->type_code();
				TYPE_LOOKUP(INSPECTOR_SELECT, dtype)
			}
		}
	}

	std::unordered_set<teq::iFunctor*> insps_;
};

#undef INSPECTOR_SELECT

}

#endif // DBG_STATS_HPP
