#include "eteq/make.hpp"

#ifdef ETEQ_MAKE_HPP

namespace eteq
{

teq::TensptrT make_layer (teq::TensptrT root,
	const std::string& layername, teq::TensptrT input)
{
	auto f = estd::must_cast<teq::iFunctor>(root.get());
	if (nullptr != f->get_attr(teq::layer_key))
	{
		teq::fatalf("attempting to attach layer attribute to node %s "
			"with an existing layer attribute", root->to_string().c_str());
	}
	f->add_attr(teq::layer_key,
		std::make_unique<teq::LayerObj>(layername, input));
	return root;
}

}

#endif
