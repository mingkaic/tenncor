#include "ead/ead.hpp"

#include "ead/dbg/interactive_dbg.hpp"

int main (int argc, char** argv)
{
	// scenario 1: REDUCE -> MUL
	{
		auto leaf = ead::make_variable_scalar<float>(0, ade::Shape({10, 3}), "leaf"); // <10,3>
		auto leaf_node = ead::convert_to_node(leaf);

		auto leaf2 = ead::make_variable_scalar<float>(0, ade::Shape({10}), "leaf2"); // <10>
		auto leaf2_node = ead::convert_to_node(leaf2);

		auto red = age::reduce_sum(leaf_node, 1); // <10>
		auto out = age::mul(red, leaf2_node); // <10>

		ead::EdgesT edges;
		auto dleaf = ead::derive_bu<float>(edges, out->get_tensor(), leaf->get_tensor()); // <10,3>

		ead::InteractiveDebugger<float> dbg;
		dbg.track(out, "out");
		dbg.edges_ = edges;
		dbg.track(dleaf, "dleaf");
		dbg.set_break();
	}

	// scenario 2: EXTEND -> MUL
	{
		auto leaf = ead::make_variable_scalar<float>(0, ade::Shape({10}), "leaf"); // <10>
		auto leaf_node = ead::convert_to_node(leaf);

		auto leaf2 = ead::make_variable_scalar<float>(0, ade::Shape({10, 3}), "leaf2"); // <10,3>
		auto leaf2_node = ead::convert_to_node(leaf2);

		auto ext = age::extend(leaf_node, 1, {3}); // <10,3>
		auto out = age::mul(ext, leaf2_node); // <10,3>

		ead::EdgesT edges;
		auto dleaf = ead::derive_bu<float>(edges, out->get_tensor(), leaf->get_tensor()); // <10>

		ead::InteractiveDebugger<float> dbg;
		dbg.track(out, "out");
		dbg.edges_ = edges;
		dbg.track(dleaf, "dleaf");
		dbg.set_break();
	}

	// scenario 3: PERMUTE -> MUL
	{
		auto leaf = ead::make_variable_scalar<float>(0, ade::Shape({3, 10}), "leaf"); // <3,10>
		auto leaf_node = ead::convert_to_node(leaf);

		auto leaf2 = ead::make_variable_scalar<float>(0, ade::Shape({10, 3}), "leaf2"); // <10,3>
		auto leaf2_node = ead::convert_to_node(leaf2);

		auto perm = age::permute(leaf_node, {1, 0}); // <10,3>
		auto out = age::mul(perm, leaf2_node); // <10,3>

		ead::EdgesT edges;
		auto dleaf = ead::derive_bu<float>(edges, out->get_tensor(), leaf->get_tensor()); // <3,10>

		ead::InteractiveDebugger<float> dbg;
		dbg.track(out, "out");
		dbg.edges_ = edges;
		dbg.track(dleaf, "dleaf");
		dbg.set_break();
	}

	// scenario 4: MATMUL -> MUL
	{
		auto leaf = ead::make_variable_scalar<float>(0, ade::Shape({3, 10}), "leaf"); // <3,10>
		auto leaf_node = ead::convert_to_node(leaf);

		auto leaf2 = ead::make_variable_scalar<float>(0, ade::Shape({5, 3}), "leaf1"); // <5,3>
		auto leaf2_node = ead::convert_to_node(leaf2);

		auto leaf3 = ead::make_variable_scalar<float>(0, ade::Shape({5, 10}), "leaf3"); // <5,10>
		auto leaf3_node = ead::convert_to_node(leaf3);

		auto perm = age::matmul(leaf_node, leaf2_node); // <5,10>
		auto out = age::mul(perm, leaf3_node); // <5,10>

		ead::EdgesT edges;
		auto dleaf = ead::derive_bu<float>(edges, out->get_tensor(), leaf->get_tensor()); // <3,10>

		ead::InteractiveDebugger<float> dbg;
		dbg.track(out, "out");
		dbg.edges_ = edges;
		dbg.track(dleaf, "dleaf");
		dbg.set_break();
	}

	// scenario 5: MATMUL -> MATMUL (separate)
	{
		auto leaf = ead::make_variable_scalar<float>(0, ade::Shape({3, 10}), "leaf"); // <3,10>
		auto leaf_node = ead::convert_to_node(leaf);

		auto leaf2 = ead::make_variable_scalar<float>(0, ade::Shape({5, 3}), "leaf1"); // <5,3>
		auto leaf2_node = ead::convert_to_node(leaf2);

		auto leaf3 = ead::make_variable_scalar<float>(0, ade::Shape({4, 5}), "leaf3"); // <4,5>
		auto leaf3_node = ead::convert_to_node(leaf3);

		auto perm = age::matmul(leaf_node, leaf2_node); // <5,10>
		auto out = age::matmul(perm, leaf3_node); // <4,10>

		ead::EdgesT edges;
		auto dleaf = ead::derive_bu<float>(edges, out->get_tensor(), leaf->get_tensor()); // <3,10>

		ead::InteractiveDebugger<float> dbg;
		dbg.track(out, "out");
		dbg.edges_ = edges;
		dbg.track(dleaf, "dleaf");
		dbg.set_break();
	}

	// scenario 6: EXTEND -> MATMUL

	// scenario 7: REDUCE -> MATMUL

	// scenario 8:
	// A<a> -> EXTEND -> B<a,b>
	// A<a> -> EXTEND -> C<a,c> -> PERMUTE -> <c,a>
	// B MATMUL C -> D<c,b>
	{
		auto leaf = ead::make_variable_scalar<float>(0, ade::Shape({5}), "leaf"); // <5>
		auto leaf_node = ead::convert_to_node(leaf);

		auto lhs = age::extend(leaf_node, 1, {10}); // <5,10>
		auto ext = age::extend(leaf_node, 1, {3}); // <5,3>

		auto rhs = age::permute(ext, {1, 0}); // <3,5>

		auto out = age::matmul(lhs, rhs); // <3,10>

		ead::EdgesT edges;
		auto dleaf = ead::derive_bu<float>(edges, out->get_tensor(), leaf->get_tensor()); // <5>

		ead::InteractiveDebugger<float> dbg;
		dbg.track(out, "out");
		dbg.edges_ = edges;
		dbg.track(dleaf, "dleaf");
		dbg.set_break();
	}

	return 0;
}
