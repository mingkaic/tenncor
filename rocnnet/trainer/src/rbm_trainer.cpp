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
			err->shape(), leaves[i].first->to_string() + "_momentum");
		auto momentum_next = discount_factor * eteq::convert_to_node(momentum) +
			(learning_rate * (1 - discount_factor) / shape_factor) * err;
		auto leaf_next = leaf_node + momentum_next;

		assigns.push_back(layr::VarAssign{
			fmts::sprintf("bbernoulli_momentum::%s_momentum_%s",
				root_label.c_str(), leaves[i].first->to_string().c_str()),
			momentum, momentum_next});
		assigns.push_back(layr::VarAssign{
			fmts::sprintf("bbernoulli_momentum::%s_grad_%s",
				root_label.c_str(), leaves[i].first->to_string().c_str()),
			leaves[i].first, leaf_next});
	}
	return {assigns};
}

layr::VarErrsT cd_grad_approx (CDChainIO& io, const layr::RBM& model,
	size_t cdk, eteq::VarptrT<PybindT> persistent)
{
	if (nullptr == io.visible_)
	{
		logs::fatal("cannot call cd_grad_approx with null visible");
	}
	if (nullptr == io.hidden_)
	{
		io.hidden_ = sample_v2h(model, io.visible_);
	}
	auto chain_it = nullptr == persistent ?
		io.hidden_ : eteq::convert_to_node(persistent);
	for (size_t i = 0; i < cdk - 1; ++i)
	{
		chain_it = gibbs_hvh(model, chain_it);
	}

	io.visible_mean_ = model.backward_connect(chain_it);
	io.hidden_mean_ = model.connect(io.visible_mean_);

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
		tenncor::matmul(tenncor::transpose(io.visible_), io.hidden_) -
		tenncor::matmul(tenncor::transpose(io.visible_mean_), io.hidden_mean_);
	layr::VarErrsT varerrs = {
		{vars[0], grad_w},
	};

	if (nullptr != vars[1])
	{
		auto grad_hb = tenncor::reduce_mean_1d(io.hidden_ - io.hidden_mean_, 1);
		varerrs.push_back({vars[1], grad_hb});
	}
	if (nullptr != vars[3])
	{
		auto grad_vb = tenncor::reduce_mean_1d(io.visible_ - io.visible_mean_, 1);
		varerrs.push_back({vars[3], grad_vb});
	}
	if (nullptr != persistent)
	{
		varerrs.push_back({persistent, gibbs_hvh(model, chain_it)});
	}
	return varerrs;
}

TrainErrF rbm_train (layr::RBM& model, eteq::iSession& sess,
	NodeptrT visible, PybindT learning_rate, PybindT discount_factor,
	ErrorF err_func, size_t cdk) // todo: add persistent option
{
	CDChainIO chain_io(visible);
	layr::VarErrsT varerrs = cd_grad_approx(chain_io, model, cdk);
	auto updates = bbernoulli_approx(varerrs, learning_rate, discount_factor);

	teq::TensptrsT to_track;
	to_track.reserve(updates.size() + 1);
	NodeptrT error = nullptr;
	if (err_func)
	{
		error = err_func(chain_io.visible_, chain_io.visible_mean_);
		to_track.push_back(error->get_tensor());
	}

	for (auto& assigns : updates)
	{
		for (auto& assign : assigns)
		{
			to_track.push_back(assign.source_->get_tensor());
		}
	}
	sess.track(to_track);

	return [&sess, updates, error]()
	{
		assign_groups_preupdate(updates,
			[&](teq::TensSetT& sources)
			{
				sess.update_target(sources);
			});
		if (nullptr == error)
		{
			return eteq::ShapedArr<PybindT>{teq::Shape(),std::vector<PybindT>{-1}};
		}
		sess.update_target({error->get_tensor().get()});
		PybindT* data = error->data();
		teq::Shape shape = error->shape();
		return eteq::ShapedArr<PybindT>{shape,
			std::vector<PybindT>(data, data + shape.n_elems()),
		};
	};
}

}

#endif // RCN_RBM_TRAINER_HPP
