#include "opt/parse.hpp"

#ifdef OPT_PARSE_HPP

namespace opt
{

static std::string to_string (::KeyVal* kv)
{
	std::string out = std::string(kv->key_) + ":[";
	auto it = kv->val_.head_;
	if (nullptr != it)
	{
		out += fmts::to_string(it->val_);
		for (it = it->next_; it != nullptr; it = it->next_)
		{
			out += "," + fmts::to_string(it->val_);
		}
	}
	out += "]";
	return out;
}

static std::string to_string (::TreeNode* node);

static std::string to_string (::Arg* arg)
{
	std::string out = to_string(arg->node_);
	auto it = arg->attrs_.head_;
	if (nullptr != it)
	{
		out += "={" + to_string((::KeyVal*) it->val_);
		for (it = it->next_; it != nullptr; it = it->next_)
		{
			out += "," + to_string((::KeyVal*) it->val_);
		}
		out += "}";
	}
	return out;
}

static std::string to_string (::Functor* functor)
{
	std::string comm_prefix;
	if (functor->commutative_)
	{
		comm_prefix = "comm ";
	}
	std::string out = comm_prefix + std::string(functor->name_) + "(";
	assert(::ARGUMENT == functor->args_.type_);
	auto it = functor->args_.head_;
	if (nullptr != it)
	{
		out += to_string((::Arg*) it->val_);
		for (it = it->next_; nullptr != it; it = it->next_)
		{
			out += "," + to_string((::Arg*) it->val_);
		}
	}

	std::string variadic(functor->variadic_);
	if (variadic.size() > 0)
	{
		variadic = ",.." + variadic;
	}
	out += variadic + ")";
	return out;
}

static std::string to_string (::TreeNode* node)
{
	std::string out;
	switch (node->type_)
	{
		case ::TreeNode::SCALAR:
			out = fmts::to_string(node->val_.scalar_);
			break;
		case ::TreeNode::ANY:
			out = std::string(node->val_.any_);
			break;
		case ::TreeNode::FUNCTOR:
			out = to_string(node->val_.functor_);
			break;
	}
	return out;
}

static std::string to_string (::Conversion* cversion)
{
	return to_string(cversion->matcher_) + "=>" + to_string(cversion->target_);
}

static std::string parse_matcher (
	std::unordered_map<std::string,MatchsT>& out,
	const ::Functor* matcher)
{
	if (NULL == matcher)
	{
		logs::fatal("cannot parse null matcher");
	}

	auto& cmatchers = out[std::string(matcher->name_)];
	MatchptrT outmatcher;
	if (matcher->commutative_)
	{
		outmatcher = std::make_shared<CommutativeMatcher>(
			std::string(matcher->variadic_));
	}
	else
	{
		outmatcher = std::make_shared<OrderedMatcher>(
			std::string(matcher->variadic_));
	}
	cmatchers.push_back(outmatcher);

	for (auto it = matcher->args_.head_; nullptr != it; it = it->next_)
	{
		auto arg = (::Arg*) it->val_;
		auto child = arg->node_;
		switch (child->type_)
		{
			case ::TreeNode::SCALAR:
				outmatcher->add_edge(new ScalarEMatcher(
					child->val_.scalar_, arg->attrs_));
				break;
			case ::TreeNode::ANY:
				outmatcher->add_edge(new AnyEMatcher(
					std::string(child->val_.any_), arg->attrs_));
				break;
			case ::TreeNode::FUNCTOR:
				outmatcher->add_edge(new FuncEMatcher(
					parse_matcher(out, child->val_.functor_),
						arg->attrs_));
				break;
			default:
				logs::fatalf("unknown matcher argument of type %d", child->type_);
		}
	}
	return outmatcher->get_fid();
}

static CversionCtx process_cversions (
	::PtrList* cversions, BuildTargetF parse_target)
{
	if (nullptr == cversions && CONVERSION != cversions->type_)
	{
		logs::fatal("rule parser did not produced conversions");
	}
	CversionCtx out;
	for (auto it = cversions->head_; it != NULL; it = it->next_)
	{
		auto cversion = (::Conversion*) it->val_;
		if (NULL == cversion)
		{
			logs::fatal("cannot parse null conversion");
		}
		auto root_id = parse_matcher(out.matchers_, cversion->matcher_);
		out.targets_.emplace(root_id, parse_target(cversion->target_));
		out.dbg_msgs_.emplace(root_id, to_string(cversion));
	}
	::cversions_free(cversions);
	return out;
}

CversionCtx parse (std::string content, BuildTargetF parse_target)
{
	::PtrList* cversions = nullptr;
	int status = ::parse_str(&cversions, content.c_str());
	if (status != 0)
	{
		logs::errorf("failed to parse content %s: got %d status",
			content.c_str(), status);
		return CversionCtx();
	}
	return process_cversions(cversions, parse_target);
}

CversionCtx parse_file (std::string filename, BuildTargetF parse_target)
{
	::PtrList* cversions = nullptr;
	FILE* file = std::fopen(filename.c_str(), "r");
	if (nullptr == file)
	{
		logs::errorf("failed to open file %s", filename.c_str());
		return CversionCtx();
	}
	int status = ::parse_file(&cversions, file);
	if (status != 0)
	{
		logs::errorf("failed to parse file %s: got %d status",
			filename.c_str(), status);
		return CversionCtx();
	}
	return process_cversions(cversions, parse_target);
}

}

#endif
