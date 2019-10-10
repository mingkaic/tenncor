#include "rocnnet/trainer/rbm_trainer.hpp"

#ifdef RCN_RBM_TRAINER_HPP

namespace trainer
{

NodeptrT sample_v2h (const layr::RBM& model, NodeptrT x)
{
	return tenncor::random::rand_binom_one(model.connect(x));
}

NodeptrT sample_h2v (const layr::RBM& model, NodeptrT x)
{
	return tenncor::random::rand_binom_one(model.backward_connect(x));
}

NodeptrT gibbs_hvh (const layr::RBM& model, NodeptrT x)
{
	return sample_v2h(model, sample_h2v(model, x));
}

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

layr::VarErrsT cd_grad_approx (layr::RBM& model, NodeptrT visible,
	size_t cdk, eteq::VarptrT<PybindT> persistent)
{
	auto hidden_sample = sample_v2h(model, visible);
	auto chain_it = nullptr == persistent ?
		hidden_sample : eteq::convert_to_node(persistent);
	for (size_t i = 0; i < cdk - 1; ++i)
	{
		chain_it = gibbs_hvh(model, chain_it);
	}

	auto visible_sample = sample_h2v(model, chain_it);
	auto hidden_reconp = model.connect(visible_sample);

	auto contents = model.get_contents();
	std::vector<eteq::VarptrT<PybindT>> vars;
	vars.reserve(contents.size());
	std::transform(contents.begin(), contents.end(),
		std::back_inserter(vars),
		[](teq::TensptrT tens) -> eteq::VarptrT<PybindT>
		{
			if (nullptr == tens)
			{
				return nullptr;
			}
			return std::make_shared<eteq::VariableNode<PybindT>>(
				std::static_pointer_cast<eteq::Variable<PybindT>>(tens));
		});

	auto grad_w =
		tenncor::matmul(tenncor::transpose(visible), hidden_sample) -
		tenncor::matmul(tenncor::transpose(visible_sample), hidden_reconp);
	layr::VarErrsT varerrs = {
		{vars[0], grad_w},
	};

	if (nullptr != vars[1])
	{
		auto grad_hb = tenncor::reduce_mean_1d(hidden_sample - hidden_reconp, 1);
		varerrs.push_back({vars[1], grad_hb});
	}
	if (nullptr != vars[3])
	{
		auto grad_vb = tenncor::reduce_mean_1d(visible - visible_sample, 1);
		varerrs.push_back({vars[3], grad_vb});
	}
	if (nullptr != persistent)
	{
		varerrs.push_back({persistent,
			sample_v2h(model, visible_sample)});
	}
	return varerrs;
}

TrainErrF bernoulli_rbm_train (layr::RBM& model, eteq::iSession& sess,
	NodeptrT visible, PybindT learning_rate, PybindT discount_factor,
	ErrorF err_func, size_t cdk)
{
	// --- todo: replace this whole section with cd_grad_approx
	auto hidden_sample = sample_v2h(model, visible);
	auto chain_it = hidden_sample; // add persistent here for pcd
	for (size_t i = 0; i < cdk - 1; ++i)
	{
		chain_it = gibbs_hvh(model, chain_it);
	}

	auto visible_sample = sample_h2v(model, chain_it);
	auto hidden_reconp = model.connect(visible_sample);

	auto grad_w =
		tenncor::matmul(tenncor::transpose(visible), hidden_sample) -
		tenncor::matmul(tenncor::transpose(visible_sample), hidden_reconp);
	auto grad_hb = tenncor::reduce_mean_1d(hidden_sample - hidden_reconp, 1);
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
	// --- end

	auto updates = bbernoulli_approx(varerrs, learning_rate, discount_factor);

	teq::TensptrsT to_track = {
		hidden_sample->get_tensor(),
		visible_sample->get_tensor(),
	};
	to_track.reserve(updates.size() + 1);
	NodeptrT error = nullptr;
	if (err_func)
	{
		error = err_func(visible, visible_sample);
		to_track.push_back(error->get_tensor());
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

	return [&sess, visible, updates, error]()
	{
		teq::TensSetT ignores = {visible->get_tensor().get()};
		assign_groups_preupdate(updates,
			[&](teq::TensSetT& sources)
			{
				sess.update_target(sources, ignores);
			});
		if (nullptr == error)
		{
			return eteq::ShapedArr<PybindT>{teq::Shape(),std::vector<PybindT>{-1}};
		}
		sess.update_target({error->get_tensor().get()}, ignores);
		PybindT* data = error->data();
		teq::Shape shape = error->shape();
		return eteq::ShapedArr<PybindT>{shape,
			std::vector<PybindT>(data, data + shape.n_elems()),
		};
	};
}

}

#endif // RCN_RBM_TRAINER_HPP
