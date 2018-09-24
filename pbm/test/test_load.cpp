#include <fstream>

#include "gtest/gtest.h"

#include "cli/util/ascii_tree.hpp"

#include "pbm/graph.hpp"


#ifndef DISABLE_LOAD_TEST


const std::string testdir = "pbm/test/data";


static inline void ltrim(std::string &s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(),
		std::not1(std::ptr_fun<int, int>(std::isspace))));
}


static inline void rtrim(std::string &s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(),
		std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}


static inline void trim(std::string &s)
{
	ltrim(s);
	rtrim(s);
}


TEST(LOAD, LoadGraph)
{
	tenncor::Graph graph;
	{
		std::fstream inputstr(testdir + "/graph.pb",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(graph.ParseFromIstream(&inputstr));
	}

	std::vector<ade::Tensorptr> roots;
	{
		std::vector<ade::Tensorptr> nodes = load_graph(graph);
		std::unordered_set<ade::iTensor*> nparents;
		for (auto it = nodes.begin(), et = nodes.end(); et != it; ++it)
		{
			if (ade::iFunctor* func = dynamic_cast<ade::iFunctor*>(it->get()))
			{
				auto refs = func->get_refs();
				for (auto ref : refs)
				{
					nparents.emplace(ref);
				}
			}
		}

		std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(roots),
		[&](ade::Tensorptr& node)
		{
			return nparents.end() == nparents.find(node.get());
		});
	}

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/graph.txt");
	ASSERT_TRUE(expectstr.is_open());
	while (std::getline(expectstr, line))
	{
		trim(line);
		if (line.size() > 0)
		{
			expect += line + "\n";
		}
	}
	for (ade::Tensorptr& root : roots)
	{
		PrettyTree<ade::iTensor*> artist(
			[](ade::iTensor*& root) -> std::vector<ade::iTensor*>
			{
				if (ade::iFunctor* f = dynamic_cast<ade::iFunctor*>(root))
				{
					return f->get_refs();
				}
				return {};
			},
			[](std::ostream& out, ade::iTensor*& root)
			{
				if (root)
				{
					if (root == ade::Tensor::SYMBOLIC_ONE.get())
					{
						out << 1;
					}
					else if (root == ade::Tensor::SYMBOLIC_ZERO.get())
					{
						out << 0;
					}
					else
					{
						out << root->to_string();
					}
				}
			});

		std::stringstream gotstr;
		artist.print(gotstr, root.get());

#if 0
		std::cout << gotstr.str() << std::endl;
#endif
		while (std::getline(gotstr, line))
		{
			trim(line);
			if (line.size() > 0)
			{
				got += line + "\n";
			}
		}
	}
	EXPECT_STREQ(expect.c_str(), got.c_str());
}


#endif /* DISABLE_LOAD_TEST */
