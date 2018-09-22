#include <functional>
#include <queue>
#include <iostream>

template <typename T>
struct PrettyTree
{
	PrettyTree (
		std::function<std::vector<T>(T&)> traverser,
		std::function<void(std::ostream&,T&)> to_stream =
		[](std::ostream& out, T v)
		{
			out << (std::string) v;
		}) :
		traverser_(traverser),
		to_stream_(to_stream) {}

	void print (std::ostream& out, T root)
	{
		out << "(";
		to_stream_(out, root);
		out << ")" << std::endl;
		std::vector<T> children = traverser_(root);
		size_t nchildren = children.size();
		if (nchildren > 0)
		{
			for (size_t i = 0; i < nchildren - 1; ++i)
			{
				out << prefix_ << " `--";
				push(" |  ");
				this->print(out, children[i]);
				pop(4);
			}
			out << prefix_ << " `--";
			push("    ");
			this->print(out, children[nchildren - 1]);
			pop(4);
		}
	}

	std::function<std::vector<T>(T&)> traverser_;

	std::function<void(std::ostream&,T&)> to_stream_;

private:
	void push (std::string sep)
	{
		prefix_ += sep;
	}

	void pop (size_t n)
	{
		if (prefix_.size() >= n)
		{
			prefix_.erase(prefix_.size() - n);
		}
		else
		{
			prefix_ = "";
		}
	}

	std::string prefix_;
};
