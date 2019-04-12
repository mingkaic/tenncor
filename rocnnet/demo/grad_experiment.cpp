#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <iostream>
#include <fstream>

#include "ead/ead.hpp"

#include "rocnnet/eqns/activations.hpp"

#include "rocnnet/trainer/mlp_trainer.hpp"

#include "ead/dbg/interactive_dbg.hpp"

// #include "ead/matcher/abstract_rep.hpp"

int main (int argc, char** argv)
{
	auto input = ead::make_variable_scalar<float>(0, ade::Shape({10, 3}), "input");
	auto expect_out = ead::make_variable_scalar<float>(0, ade::Shape({5, 3}), "expect_out");

	auto weight0 = ead::make_variable_scalar<float>(1, ade::Shape({9, 10}), "weight0");
	auto bias0 = ead::make_variable_scalar<float>(0, ade::Shape({9}), "bias0");
	auto weight1 = ead::make_variable_scalar<float>(1, ade::Shape({5, 9}), "weight1");
	auto bias1 = ead::make_variable_scalar<float>(0, ade::Shape({5}), "bias1");

	auto input_node = ead::convert_to_node(input);
	auto expect_out_node = ead::convert_to_node(expect_out);
	auto w0_node = ead::convert_to_node(weight0);
	auto b0_node = ead::convert_to_node(bias0);
	auto w1_node = ead::convert_to_node(weight1);
	auto b1_node = ead::convert_to_node(bias1);

	auto input_matmul = age::matmul(input_node, w0_node);
	auto layer0 = eqns::weighed_bias_add(input_matmul, b0_node);
	auto sig0 = age::sigmoid(layer0);
	auto sig0_matmul = age::matmul(sig0, w1_node);
	auto layer1 = eqns::weighed_bias_add(sig0_matmul, b1_node);
	auto sig1 = age::sigmoid(layer1);
	auto diff = age::sub(expect_out_node, sig1);
	auto error = age::square(diff);

	// wrt to w0
	ead::EdgesT edges;
	auto derror_w0 = ead::derive_with_edges<float>(edges, error, w0_node);
	auto ddiff_w0 = ead::derive_with_edges<float>(edges, diff, w0_node);
	auto dsig1_w0 = ead::derive_with_edges<float>(edges, sig1, w0_node);
	auto dlayer1_w0 = ead::derive_with_edges<float>(edges, layer1, w0_node);
	auto dsig0_matmul_w0 = ead::derive_with_edges<float>(edges, sig0_matmul, w0_node);
	auto dsig0_w0 = ead::derive_with_edges<float>(edges, sig0, w0_node);
	auto dlayer0_w0 = ead::derive_with_edges<float>(edges, layer0, w0_node);
	auto dinput_matmul_w0 = ead::derive_with_edges<float>(edges, input_matmul, w0_node);

	ead::NodesT<float> w0s = {
		derror_w0,
		ddiff_w0,
		dsig1_w0,
		dlayer1_w0,
		dsig0_matmul_w0,
		dsig0_w0,
		dlayer0_w0,
		dinput_matmul_w0,
		error,
	};

	ade::TensT tensors;
	tensors.reserve(w0s.size());
	std::transform(w0s.begin(), w0s.end(), std::back_inserter(tensors),
		[](ead::NodeptrT<float>& node) { return node->get_tensor(); });
	// auto graph = ead::represent(tensors);
	// auto tens_out = ead::actualize<float>(graph);
	// auto wrt_w0 = ead::ops_reuse<float>(
	// 	ead::multi_optimize<float>(ead::ops_reuse<float>(tens_out)));

	auto wrt_w0 = ead::ops_reuse<float>(ead::multi_optimize<float>(w0s));

	// derror_w0 = wrt_w0[0];
	// ddiff_w0 = wrt_w0[1];
	// dsig1_w0 = wrt_w0[2];
	// dlayer1_w0 = wrt_w0[3];
	// dsig0_matmul_w0 = wrt_w0[4];
	// dsig0_w0 = wrt_w0[5];
	// dlayer0_w0 = wrt_w0[6];
	// dinput_matmul_w0 = wrt_w0[7];
	// error = wrt_w0[8];

	ead::InteractiveDebugger<float> dbg;
	dbg.track(error, "error");
	dbg.edges_ = edges;

	// {
	// 	auto root = static_cast<ade::iFunctor*>(dinput_matmul_w0->get_tensor().get());
	// 	auto perm = static_cast<ade::iFunctor*>(root->get_children()[0].get_tensor().get());
	// 	dbg.edges_.push_back(
	// 		ead::Edge{
	// 			perm->get_children()[0].get_tensor(),
	// 			input_matmul->get_tensor(),
	// 			ade::Opcode{
	// 				"JACOBIAN_input_matmul",
	// 				ead::GRADIENT
	// 			}
	// 		}
	// 	);
	// 	dbg.edges_.push_back(
	// 		ead::Edge{
	// 			perm->get_children()[0].get_tensor(),
	// 			layer0->get_tensor(),
	// 			ade::Opcode{
	// 				"JACOBIAN_layer0_w0",
	// 				ead::GRADIENT
	// 			}
	// 		}
	// 	);
	// }
	// {
	// 	auto root = static_cast<ade::iFunctor*>(dsig0_w0->get_tensor().get());
	// 	auto perm = static_cast<ade::iFunctor*>(root->get_children()[0].get_tensor().get());
	// 	auto mul = static_cast<ade::iFunctor*>(perm->get_children()[0].get_tensor().get());
	// 	auto ext = static_cast<ade::iFunctor*>(mul->get_children()[1].get_tensor().get());
	// 	dbg.edges_.push_back(
	// 		ead::Edge{
	// 			ext->get_children()[0].get_tensor(),
	// 			sig0->get_tensor(),
	// 			ade::Opcode{
	// 				"JACOBIAN_sig0_w0",
	// 				ead::GRADIENT
	// 			}
	// 		}
	// 	);
	// }
	// {
	// 	auto root = static_cast<ade::iFunctor*>(dsig0_matmul_w0->get_tensor().get());
	// 	auto perm = static_cast<ade::iFunctor*>(root->get_children()[0].get_tensor().get());
	// 	auto mul = static_cast<ade::iFunctor*>(perm->get_children()[0].get_tensor().get());
	// 	auto ext = static_cast<ade::iFunctor*>(mul->get_children()[1].get_tensor().get());
	// 	auto mul2 = static_cast<ade::iFunctor*>(ext->get_children()[0].get_tensor().get());
	// 	auto red = static_cast<ade::iFunctor*>(mul2->get_children()[1].get_tensor().get());
	// 	auto perm2 = static_cast<ade::iFunctor*>(red->get_children()[0].get_tensor().get());
	// 	dbg.edges_.push_back(
	// 		ead::Edge{
	// 			perm2->get_children()[0].get_tensor(),
	// 			sig0_matmul->get_tensor(),
	// 			ade::Opcode{
	// 				"JACOBIAN_sig0_matmul_w0",
	// 				ead::GRADIENT
	// 			}
	// 		}
	// 	);
	// 	dbg.edges_.push_back(
	// 		ead::Edge{
	// 			perm2->get_children()[0].get_tensor(),
	// 			layer1->get_tensor(),
	// 			ade::Opcode{
	// 				"JACOBIAN_layer1_w0",
	// 				ead::GRADIENT
	// 			}
	// 		}
	// 	);
	// }
	// {
	// 	auto root = static_cast<ade::iFunctor*>(dsig1_w0->get_tensor().get());
	// 	auto perm = static_cast<ade::iFunctor*>(root->get_children()[0].get_tensor().get());
	// 	auto mul = static_cast<ade::iFunctor*>(perm->get_children()[0].get_tensor().get());
	// 	auto ext = static_cast<ade::iFunctor*>(mul->get_children()[1].get_tensor().get());
	// 	auto mul2 = static_cast<ade::iFunctor*>(ext->get_children()[0].get_tensor().get());
	// 	auto red = static_cast<ade::iFunctor*>(mul2->get_children()[1].get_tensor().get());
	// 	auto perm2 = static_cast<ade::iFunctor*>(red->get_children()[0].get_tensor().get());
	// 	auto mul3 = static_cast<ade::iFunctor*>(perm2->get_children()[0].get_tensor().get());
	// 	auto ext2 = static_cast<ade::iFunctor*>(mul3->get_children()[1].get_tensor().get());
	// 	dbg.edges_.push_back(
	// 		ead::Edge{
	// 			ext2->get_children()[0].get_tensor(),
	// 			sig1->get_tensor(),
	// 			ade::Opcode{
	// 				"JACOBIAN_sig1_w0",
	// 				ead::GRADIENT
	// 			}
	// 		}
	// 	);
	// }
	// {
	// 	auto root = static_cast<ade::iFunctor*>(ddiff_w0->get_tensor().get());
	// 	auto perm = static_cast<ade::iFunctor*>(root->get_children()[0].get_tensor().get());
	// 	auto mul = static_cast<ade::iFunctor*>(perm->get_children()[0].get_tensor().get());
	// 	auto ext = static_cast<ade::iFunctor*>(mul->get_children()[1].get_tensor().get());
	// 	auto mul2 = static_cast<ade::iFunctor*>(ext->get_children()[0].get_tensor().get());
	// 	auto red = static_cast<ade::iFunctor*>(mul2->get_children()[1].get_tensor().get());
	// 	auto perm2 = static_cast<ade::iFunctor*>(red->get_children()[0].get_tensor().get());
	// 	auto mul3 = static_cast<ade::iFunctor*>(perm2->get_children()[0].get_tensor().get());
	// 	auto ext2 = static_cast<ade::iFunctor*>(mul3->get_children()[1].get_tensor().get());
	// 	auto mul4 = static_cast<ade::iFunctor*>(ext2->get_children()[0].get_tensor().get());
	// 	dbg.edges_.push_back(
	// 		ead::Edge{
	// 			mul4->get_children()[1].get_tensor(),
	// 			diff->get_tensor(),
	// 			ade::Opcode{
	// 				"JACOBIAN_diff_w0",
	// 				ead::GRADIENT
	// 			}
	// 		}
	// 	);
	// }
	// {
	// 	auto root = static_cast<ade::iFunctor*>(derror_w0->get_tensor().get());
	// 	auto perm = static_cast<ade::iFunctor*>(root->get_children()[0].get_tensor().get());
	// 	auto mul = static_cast<ade::iFunctor*>(perm->get_children()[0].get_tensor().get());
	// 	auto ext = static_cast<ade::iFunctor*>(mul->get_children()[1].get_tensor().get());
	// 	auto mul2 = static_cast<ade::iFunctor*>(ext->get_children()[0].get_tensor().get());
	// 	auto red = static_cast<ade::iFunctor*>(mul2->get_children()[1].get_tensor().get());
	// 	auto perm2 = static_cast<ade::iFunctor*>(red->get_children()[0].get_tensor().get());
	// 	auto mul3 = static_cast<ade::iFunctor*>(perm2->get_children()[0].get_tensor().get());
	// 	auto ext2 = static_cast<ade::iFunctor*>(mul3->get_children()[1].get_tensor().get());
	// 	auto mul4 = static_cast<ade::iFunctor*>(ext2->get_children()[0].get_tensor().get());
	// 	auto mul5 = static_cast<ade::iFunctor*>(mul4->get_children()[1].get_tensor().get());
	// 	dbg.edges_.push_back(
	// 		ead::Edge{
	// 			mul5->get_children()[1].get_tensor(),
	// 			error->get_tensor(),
	// 			ade::Opcode{
	// 				"JACOBIAN_error_w0",
	// 				ead::GRADIENT
	// 			}
	// 		}
	// 	);
	// }

	// dbg.track(dinput_matmul_w0, "dinput_matmul_w0");
	// dbg.set_break();
	// dbg.track(dlayer0_w0, "dlayer0_w0");
	// dbg.set_break();
	// dbg.track(dsig0_w0, "dsig0_w0");
	// dbg.set_break();
	// dbg.track(dsig0_matmul_w0, "dsig0_matmul_w0");
	// dbg.set_break();
	// dbg.track(dlayer1_w0, "dlayer1_w0");
	// dbg.set_break();
	// dbg.track(dsig1_w0, "dsig1_w0");
	// dbg.set_break();
	// dbg.track(ddiff_w0, "ddiff_w0");
	// dbg.set_break();
	dbg.track(derror_w0, "derror_w0");
	dbg.set_break();

	return 0;
}
