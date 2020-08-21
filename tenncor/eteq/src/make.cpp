#include "tenncor/eteq/make.hpp"

#ifdef ETEQ_MAKE_HPP

namespace eteq
{

teq::TensptrT add_dependencies (teq::TensptrT root,
	teq::TensptrsT dependencies)
{
	auto f = estd::must_cast<teq::iFunctor>(root.get());
	auto deps_attr = dynamic_cast<teq::TensArrayT*>(
		f->get_attr(dependency_key));
	if (nullptr == deps_attr)
	{
		f->add_attr(dependency_key,
			std::make_unique<teq::TensArrayT>());
		deps_attr = static_cast<teq::TensArrayT*>(
			f->get_attr(dependency_key));
	}
	auto& contents = deps_attr->contents_;
	for (auto& tens : dependencies)
	{
		contents.emplace(contents.end(),
			std::make_unique<teq::TensorObj>(tens));
	}
	return root;
}

}

#endif
