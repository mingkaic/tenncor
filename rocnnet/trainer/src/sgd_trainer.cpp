#include "rocnnet/trainer/sgd_trainer.hpp"

#ifdef RCN_SGD_TRAINER_HPP

namespace trainer
{

TrainErrF sgd_train (layr::iLayer& model, teq::iSession& sess,
	LinkptrT train_in, LinkptrT expected_out, layr::ApproxF update,
	layr::ErrorF errfunc, NodeUnarF gradprocess)
{
	auto train_out = model.connect(train_in);
	auto error = errfunc(expected_out, train_out);

	auto contents = model.get_contents();
	layr::VarErrsT vars;
	for (auto tens : contents)
	{
		if (auto var = std::dynamic_pointer_cast<
			eteq::Variable<PybindT>>(tens))
		{
			vars.push_back({var, gradprocess(
				eteq::derive(error, eteq::to_node<PybindT>(var)))});
		}
	}
	auto updates = update(vars);

	teq::TensptrsT track_batch = {
		train_out->get_tensor(),
		error->get_tensor(),
	};
	for (layr::AssignsT& assigns : updates)
	{
		for (layr::VarAssign& assign : assigns)
		{
			track_batch.push_back(assign.source_->get_tensor());
		}
	}
	sess.track(track_batch);

	return [&sess, updates, error](void)
	{
		assign_groups_preupdate(updates,
			[&](teq::TensSetT& sources)
			{
				sess.update_target(sources);
			});
		sess.update_target({error->get_tensor().get()});
		PybindT* data = error->data();
		teq::Shape shape = error->shape();
		return teq::ShapedArr<PybindT>{shape,
			std::vector<PybindT>(data, data + shape.n_elems()),
		};
	};
}

}

#endif
