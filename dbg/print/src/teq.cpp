#include "dbg/print/teq.hpp"

#ifdef DBG_TEQ_HPP

void ten_stream (std::ostream& out,
	teq::iTensor*& root, const std::string&,
	const PrintEqConfig& cfg)
{
	if (nullptr == root)
	{
		return;
	}
	auto it = cfg.labels_.find(root);
	if (cfg.labels_.end() != it)
	{
		out << it->second << "=";
	}
	if (auto var = dynamic_cast<teq::iLeaf*>(root))
	{
		out << teq::get_usage_name(var->get_usage()) << ":";
	}
	out << root->to_string();
	if (cfg.showtype_)
	{
		out << "<" << root->get_meta().type_label() << ">";
	}
	if (cfg.showshape_)
	{
		out << root->shape().old_string();
	}
	if (cfg.showvers_)
	{
		out << ":version=" << root->get_meta().state_version();
	}
	if (auto fnc = dynamic_cast<teq::iFunctor*>(root))
	{
		if (cfg.lsattrs_)
		{
			auto attrs = fnc->ls_attrs();
			out << ":attrkeys=" << fmts::to_string(attrs.begin(), attrs.end());
		}
		if (auto attr = fnc->get_attr(cfg.showattr_))
		{
			out << ":attr=" << attr->to_string();
		}
	}
}

#endif
