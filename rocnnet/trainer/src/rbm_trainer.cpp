#include "rocnnet/trainer/rbm_trainer.hpp"

#ifdef RCN_RBM_TRAINER_HPP

namespace trainer
{

layr::AssignGroupsT bbernoulli_approx (const layr::VarErrsT& leaves,
	PybindT learning_rate, PybindT discount_factor, std::string root_label)
{
	// assign momentums before leaves
	layr::AssignsT assigns;
	for (size_t i = 0, nleaves = leaves.size(); i < nleaves; ++i)
	{
		auto leaf_node = eteq::convert_to_node(leaves[i].first);
		auto err = leaves[i].second;

		auto shape = err->shape();
		std::vector<teq::DimT> slist(shape.begin(), shape.end());
		auto it = slist.rbegin(), et = slist.rend();
		while (it != et && *it == 1)
		{
			++it;
		}
		teq::DimT shape_factor = it == et ? 1 : *it;
		auto momentum = eteq::make_variable_scalar<PybindT>(0,
			err->shape(), leaves[i].first->get_label() + "_momentum");
		auto momentum_next = discount_factor * eteq::convert_to_node(momentum) +
			(learning_rate * (1 - discount_factor) / shape_factor) * err;
		auto leaf_next = leaf_node + momentum_next;

		assigns.push_back(layr::VarAssign{
			fmts::sprintf("bbernoulli_momentum::%s_momentum_%s",
				root_label.c_str(), leaves[i].first->get_label().c_str()),
			momentum, momentum_next});
		assigns.push_back(layr::VarAssign{
			fmts::sprintf("bbernoulli_momentum::%s_grad_%s",
				root_label.c_str(), leaves[i].first->get_label().c_str()),
			leaves[i].first, leaf_next});
	}
	return {assigns};
}

TrainErrF bernoulli_rbm_train (layr::RBM& model, eteq::iSession& sess,
	NodeptrT visible, PybindT learning_rate, PybindT discount_factor,
	ErrorF err_func)
{
	auto hidden_sample = model.connect(visible);
	auto visible_sample = model.backward_connect(
		tenncor::random::rand_binom_one(hidden_sample));

	auto hidden_reconp = model.connect(visible_sample);

	auto grad_w =
		tenncor::matmul(tenncor::transpose(visible), hidden_sample) -
		tenncor::matmul(tenncor::transpose(visible_sample), hidden_reconp);
	auto grad_hb = tenncor::reduce_mean_1d(
		hidden_sample - hidden_reconp, 1);
	auto grad_vb = tenncor::reduce_mean_1d(visible - visible_sample, 1);

	auto contents = model.get_contents();
	std::vector<eteq::VarptrT<PybindT>> vars;
	vars.reserve(contents.size());
	std::transform(contents.begin(), contents.end(),
		std::back_inserter(vars),
		[](teq::TensptrT tens)
		{
			return std::make_shared<eteq::VariableNode<PybindT>>(
				std::static_pointer_cast<eteq::Variable<PybindT>>(tens));
		});
	layr::VarErrsT varerrs = {
		{vars[0], grad_w},
		{vars[1], grad_hb},
		{vars[3], grad_vb},
	};

	auto updates = bbernoulli_approx(varerrs, learning_rate, discount_factor);

	teq::TensptrsT to_track = {
		hidden_sample->get_tensor(),
		visible_sample->get_tensor(),
	};
	to_track.reserve(updates.size() + 1);
	if (err_func)
	{
		error_ = err_func(visible, visible_sample);
		to_track.push_back(error_->get_tensor());
	}

	for (auto& assigns : updates)
	{
		for (auto& assign : assigns)
		{
			auto source = assign.source_->get_tensor();
			to_track.push_back(source);
		}
	}
	sess.track(to_track);

	return [&sess, assign_sources, visible, updates, error]()
	{
		teq::TensSetT ignores = {visible->get_tensor().get()};
		assign_groups_preupdate(updates,
			[&](teq::TensSetT& sources)
			{
				sess.update_targeted(sources, ignores);
			});
		if (nullptr == error)
		{
			return eteq::ShapedArr<PybindT>{teq::Shape(),std::vector<PybindT>{-1}};
		}
		sess.update_targeted({error->get_tensor().get()}, ignores);
		PybindT* data = error->data();
		teq::Shape shape = error->shape();
		return eteq::ShapedArr<PybindT>{shape,
			std::vector<PybindT>(data, data + shape.n_elems()),
		};
	};
}

}

#endif // RCN_RBM_TRAINER_HPP
