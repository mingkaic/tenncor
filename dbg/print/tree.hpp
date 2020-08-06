///
/// tree.hpp
/// dbg
///
/// Purpose:
/// Draw a generic tree/graph structure as an ascii tree
///

#include <functional>
#include <iostream>
#include <string>

#ifndef DBG_TREE_HPP
#define DBG_TREE_HPP

const char vert_branch = '-';

const char default_indent = '_';

/// Generically draw a traversible structure
/// Traversible structure is anything having a parent-children relationship
/// Template argument T defines one node of the traversible structure
template <typename T>
struct PrettyTree final
{
	/// Traverser and streamer methods define behavior for traversing through
	/// and displaying elements of a generic structure
	PrettyTree (std::function<std::vector<T>(T&,size_t)> traverser,
		std::function<void(std::ostream&,T&,const std::string&)> to_stream) :
		traverser_(traverser), to_stream_(to_stream) {}

	/// Given the output stream, and a root to start the traversal,
	/// wrap brackets around string representation of each node and connect
	/// each node with series of | and `-- branches and stream to out
	void print (std::ostream& out, T root)
	{
		size_t depth = 0;
		print_helper(out, root, "", depth);
	}

	/// Horizontal length of the branch, by default the branch looks like `--
	size_t branch_length_ = 2;

	/// Behavior of traversing through a structure
	std::function<std::vector<T>(T&,size_t)> traverser_;

	/// Behavior of displaying a node in the structure
	std::function<void(std::ostream&,T&,const std::string&)> to_stream_;

	char indent_ = default_indent;

	std::pair<std::string,std::string> node_wrap = {"(", ")"};

private:
	void print_helper (std::ostream& out, T root,
		const std::string& prefix, size_t depth)
	{
		out << node_wrap.first;
		to_stream_(out, root, prefix);
		out << node_wrap.second << "\n";
		std::vector<T> children = traverser_(root, depth);
		size_t nchildren = children.size();
		if (nchildren > 0)
		{
			std::string branch = prefix + indent_ + "`" +
				std::string(branch_length_, vert_branch);
			for (size_t i = 0; i < nchildren - 1; ++i)
			{
				out << branch;
				this->print_helper(out, children[i], prefix + indent_ + "|" +
					std::string(branch_length_, indent_), depth + 1);
			}
			out << branch;
			this->print_helper(out, children[nchildren - 1],
				prefix + std::string(2 + branch_length_, indent_), depth + 1);
		}
	}
};

#endif // DBG_TREE_HPP
