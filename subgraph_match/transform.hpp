#include "subgraph_match/irep.hpp"

#ifndef OPT_TRANSFORM_HPP
#define OPT_TRANSFORM_HPP

namespace opt
{

struct Transform final
{
	Transform (std::string stmt)
	{
		auto parts = fmts::split(stmt, tuple_delim);
		if (parts.size() < 2)
		{
			logs::fatalf(
				"statement must consist at least 2 parts delimited by %s: %s",
				tuple_delim.c_str(), stmt.c_str());
		}
		stop_ = parts.size() == 2;
		std::string transform_half = parts[0];
		std::string result_half = parts[1];
		fmts::trim(transform_half);
		fmts::trim(result_half);
		pattern_ = std::regex(transform_half);
		simplification_ = result_half;
		pheight_ = fmts::split(transform_half, line_delim).size();
	}

	bool simplify (TokenptrT& root, IdTokenMapT& token_map)
	{
		std::string serial = root->encode(pheight_);
		bool matched = std::regex_match(serial, pattern_);
		if (matched)
		{
			std::string replacement = std::regex_replace(
				serial, pattern_, simplification_);
			auto levels = fmts::split(std::regex_replace(serial, pattern_,
				simplification_), line_delim);

			root = std::make_shared<TokenNode>(levels[0]);
			std::list<TokenptrT> parents = {root};
			for (size_t i = 1, n = levels.size(); i < n; ++i)
			{
				std::string level = levels[i];
				fmts::trim(level);
				std::vector<std::string> tokens = fmts::split(level, ",");
				std::list<TokenptrT> new_parents;
				for (std::string token : tokens)
				{
					fmts::trim(token);
					if (token.empty())
					{
						continue;
					}
					auto it = parents.begin();
					assert(parents.end() != it);
					auto node = std::make_shared<TokenNode>(token);
					if (nullptr == node)
					{
						logs::fatalf("failed to parse token: '%s'",
							token.c_str());
					}
					TokenptrT parent = *it;
					parent->children_.push_back(node);
					if (parent->children_.size() == parent->nallowed_children_)
					{
						parents.pop_front();
					}
					if (node->nallowed_children_ > 0)
					{
						new_parents.push_back(node);
					}
				}
				parents = new_parents;
			}
			for (TokenptrT& parent : parents)
			{
				auto it = token_map.find(parent->tens_id_);
				if (token_map.end() == it)
				{
					logs::fatalf("cannot match unknown parent token %s",
						parent->to_string().c_str());
				}
				parent->children_ = it->second->children_;
			}
		}
		return matched && stop_;
	}

	bool stop_; // whether to stop if we hit something matches pattern

	std::regex pattern_;

	std::string simplification_;

	size_t pheight_;
};

using TransformsT = std::vector<Transform>;

}

#endif // OPT_TRANSFORM_HPP
