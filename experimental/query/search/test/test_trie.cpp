
#ifndef DISABLE_TEST_TRIE_HPP


#include <iostream>
#include <string>
#include <sstream>

#include "gtest/gtest.h"

#include "experimental/query/search/trie.hpp"


const size_t nalphabet = 'z' - 'a' + 1;


struct AlphaHash final
{
	size_t operator() (char c) const
	{
		int idx = c == ' ' ? 0 : std::tolower(c) - 'a' + 1;
		if (idx < 0 || idx > nalphabet)
		{
			std::stringstream ss;
			ss << "unexpected character '" << c << "'";
			throw std::runtime_error(ss.str());
		}
		return idx;
	}
};


struct Empty final {};


using TrieT = estd::Trie<std::string,Empty,AlphaHash>;


TEST(TRIE, TrieContains)
{
	TrieT t;
	t.add("hello my darlin", Empty());
	t.add("hello my darin", Empty());

	EXPECT_FALSE(t.contains_exact("hello"));
	EXPECT_FALSE(t.contains_exact("hello my daryn"));
	EXPECT_FALSE(t.contains_exact("hello my darie"));
	EXPECT_TRUE(t.contains_exact("hello my darlin"));
	EXPECT_TRUE(t.contains_exact("hello my darin"));

	EXPECT_TRUE(t.contains_prefix("hello"));
	EXPECT_FALSE(t.contains_prefix("hello my dary"));
	EXPECT_FALSE(t.contains_prefix("hello my darie"));
	EXPECT_TRUE(t.contains_prefix("hello my darlin"));
	EXPECT_TRUE(t.contains_prefix("hello my darin"));
}


TEST(TRIE, TrieBestPrefix)
{
	TrieT t;
	t.add("hello my honey", Empty());
	t.add("hello my hunee", Empty());

	EXPECT_STREQ("hello", t.best_prefix("hello").c_str());
	EXPECT_STREQ("hello my ", t.best_prefix("hello my ragtime").c_str());
	EXPECT_STREQ("", t.best_prefix("summertime gal").c_str());
	EXPECT_STREQ("h", t.best_prefix("honey you lose me").c_str());
	EXPECT_STREQ("hello my hunee", t.best_prefix("hello my huneebun").c_str());
}


TEST(TRIE, TrieRemove)
{
	TrieT t;
	t.add("send me a kiss on wire", Empty());
	t.add("send me a kiss on wire on wire", Empty());
	t.add("sent ma heart on fire", Empty());
	t.add("sent ma heart on fyre", Empty());

	t.rm("send");
	EXPECT_TRUE(t.contains_exact("send me a kiss on wire"));
	EXPECT_TRUE(t.contains_exact("send me a kiss on wire on wire"));
	EXPECT_TRUE(t.contains_exact("sent ma heart on fire"));
	EXPECT_TRUE(t.contains_exact("sent ma heart on fyre"));

	t.rm("send me a kiss on wire");
	EXPECT_FALSE(t.contains_exact("send me a kiss on wire"));
	EXPECT_TRUE(t.contains_exact("send me a kiss on wire on wire"));
	EXPECT_TRUE(t.contains_exact("sent ma heart on fire"));
	EXPECT_TRUE(t.contains_exact("sent ma heart on fyre"));

	t.rm("sent ma heart on fyre");
	EXPECT_TRUE(t.contains_exact("send me a kiss on wire on wire"));
	EXPECT_TRUE(t.contains_exact("sent ma heart on fire"));
	EXPECT_FALSE(t.contains_exact("sent ma heart on fyre"));

	t.rm("sen");
	EXPECT_TRUE(t.contains_exact("send me a kiss on wire on wire"));
	EXPECT_TRUE(t.contains_exact("sent ma heart on fire"));

	t.rm("send me a kiss on wire on wire");
	EXPECT_FALSE(t.contains_exact("send me a kiss on wire on wire"));
	EXPECT_TRUE(t.contains_exact("sent ma heart on fire"));

	t.rm("sent ma heart on fire");
	EXPECT_FALSE(t.contains_exact("sent ma heart on fire"));
}


#endif // DISABLE_TEST_TRIE_HPP
