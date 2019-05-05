
#ifndef DISABLE_EDGE_TEST


#include <unordered_set>

#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "ade/test/common.hpp"

#include "ade/edge.hpp"


TEST(EDGE, Expiration)
{
	std::vector<ade::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	ade::Opcode mock_code{"MOCK_EDGE", 435};
	ade::Shape shape(slist);

	ade::TensptrT parent(new MockTensor(shape));
	ade::TensptrT child(new MockTensor(shape));

	ade::Edge edge;
	{
		ade::TensptrT tempparent(new MockTensor(shape));
		ade::TensptrT tempchild(new MockTensor(shape));
		edge = ade::Edge{tempparent, tempchild, mock_code};
		EXPECT_FALSE(edge.expired());
	}
	EXPECT_TRUE(edge.expired());

	{
		ade::TensptrT tempparent(new MockTensor(shape));
		edge = ade::Edge{tempparent, child, mock_code};
		EXPECT_FALSE(edge.expired());
	}
	EXPECT_TRUE(edge.expired());

	{
		ade::TensptrT tempchild(new MockTensor(shape));
		edge = ade::Edge{parent, tempchild, mock_code};
		EXPECT_FALSE(edge.expired());
	}
	EXPECT_TRUE(edge.expired());
}


TEST(EDGE, Equality)
{
	std::vector<ade::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	ade::Opcode mock_code{"MOCK_EDGE", 435};
	ade::Opcode mock_code2{"MOCK_EDGE2", 436};
	ade::Shape shape(slist);

	ade::TensptrT parent(new MockTensor(shape));
	ade::TensptrT parent2(new MockTensor(shape));
	ade::TensptrT child(new MockTensor(shape));
	ade::TensptrT child2(new MockTensor(shape));

	ade::Edge orig_edge{parent, child, mock_code};
	ade::Edge edge{parent2, child, mock_code};
	ade::Edge edge2{parent, child2, mock_code};
	ade::Edge edge3{parent2, child2, mock_code};
	ade::Edge edge4{parent, child, mock_code2};

	ade::Edge expired_edge;
	ASSERT_TRUE(expired_edge.expired());

	ade::Edge edge_eq{parent, child, mock_code};

	EXPECT_FALSE(orig_edge == edge);
	EXPECT_FALSE(orig_edge == edge2);
	EXPECT_FALSE(orig_edge == edge3);
	EXPECT_FALSE(orig_edge == edge4);
	EXPECT_FALSE(orig_edge == expired_edge);
	EXPECT_TRUE(orig_edge == edge_eq);
}


TEST(EDGE, Hash)
{
	std::vector<ade::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	ade::Opcode mock_code{"MOCK_EDGE", 435};
	ade::Opcode mock_code2{"MOCK_EDGE2", 436};
	ade::Shape shape(slist);

	ade::TensptrT parent(new MockTensor(shape));
	ade::TensptrT parent2(new MockTensor(shape));
	ade::TensptrT child(new MockTensor(shape));
	ade::TensptrT child2(new MockTensor(shape));

	ade::Edge expired_edge;
	ade::Edge expired_edge2 = {
		ade::TensrefT(),
		ade::TensrefT(),
		ade::Opcode{"SOMETHING", 123},
	};
	ASSERT_TRUE(expired_edge.expired());
	ASSERT_TRUE(expired_edge2.expired());

	std::unordered_set<ade::Edge,ade::EdgeHash> edges = {
		ade::Edge{parent, child, mock_code},
		ade::Edge{parent2, child, mock_code},
		ade::Edge{parent, child2, mock_code},
		ade::Edge{parent2, child2, mock_code},
		ade::Edge{parent, child, mock_code2},
		ade::Edge{parent, child, mock_code},
		expired_edge,
		expired_edge2,
	};

	EXPECT_EQ(6, edges.size());
}


#endif // DISABLE_EDGE_TEST
