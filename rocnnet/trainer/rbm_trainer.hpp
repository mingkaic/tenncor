#include "rocnnet/modl/rbm.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#ifndef MODL_RBM_TRAINER_HPP
#define MODL_RBM_TRAINER_HPP

namespace trainer
{

// Bernoulli RBM "error approximation"
// for each (x, err) in leaves
// momentum_next ~ χ * momentum_cur + η * (1 - χ) / err.shape[0] * err
// x_next = x_curr + next_momentum
//
// where η is the learning rate, and χ is discount_factor
eqns::AssignGroupsT bbernoulli_approx (const eqns::VarErrsT& leaves,
	PybindT learning_rate, PybindT discount_factor,
	std::string root_label = "")
{
	// assign momentums before leaves
	eqns::AssignsT assigns;
	for (size_t i = 0, nleaves = leaves.size(); i < nleaves; ++i)
	{
		auto leaf_node = ead::convert_to_node(leaves[i].first);
		auto err = leaves[i].second;

		auto shape = err->shape();
		std::vector<ade::DimT> slist(shape.begin(), shape.end());
		auto it = slist.rbegin(), et = slist.rend();
		while (it != et && *it == 1)
		{
			++it;
		}
		ade::DimT shape_factor = it == et ? 1 : *it;
		auto momentum = ead::make_variable_scalar<PybindT>(0,
			err->shape(), leaves[i].first->get_label() + "_momentum");
		auto momentum_next = tenncor::add(
			tenncor::mul(
				ead::make_constant_scalar(discount_factor, momentum->shape()),
				ead::convert_to_node(momentum)),
			tenncor::mul(
				ead::make_constant_scalar(learning_rate *
					(1 - discount_factor) / shape_factor, err->shape()),
				err));
		auto leaf_next = tenncor::add(leaf_node, momentum_next);

		assigns.push_back(eqns::VarAssign{
			fmts::sprintf("bbernoulli_momentum::%s_momentum_%s",
				root_label.c_str(), leaves[i].first->get_label().c_str()),
			momentum, momentum_next});
		assigns.push_back(eqns::VarAssign{
			fmts::sprintf("bbernoulli_momentum::%s_grad_%s",
				root_label.c_str(), leaves[i].first->get_label().c_str()),
			leaves[i].first, leaf_next});
	}
	return {assigns};
}

using ErrorF = std::function<ead::NodeptrT<PybindT>(ead::NodeptrT<PybindT>,ead::NodeptrT<PybindT>)>;

struct BernoulliRBMTrainer final
{
	BernoulliRBMTrainer (modl::RBM& model,
		ead::iSession& sess,
		ade::DimT batch_size,
		PybindT learning_rate,
		PybindT discount_factor,
		ErrorF err_func = ErrorF()) :
		model_(model), sess_(&sess), batch_size_(batch_size)
	{
		visible_ = ead::make_variable_scalar<PybindT>(0,
			ade::Shape({(ade::DimT) model.get_ninput(), batch_size}));

		hidden_sample_ = model.connect(visible_);
		visible_sample_ = model.backward_connect(
			tenncor::random::rand_binom_one(hidden_sample_));

		auto hidden_reconp = model.connect(visible_sample_);

		auto grad_w = tenncor::sub(
			tenncor::matmul(tenncor::transpose(
				ead::convert_to_node(visible_)), hidden_sample_),
			tenncor::matmul(tenncor::transpose(
				visible_sample_), hidden_reconp));
		auto grad_hb = tenncor::reduce_mean_1d(
			tenncor::sub(hidden_sample_, hidden_reconp), 1);
		auto grad_vb = tenncor::reduce_mean_1d(
			tenncor::sub(ead::convert_to_node(visible_), visible_sample_), 1);

		auto contents = model.get_contents();
		std::vector<ead::VarptrT<PybindT>> vars;
		vars.reserve(contents.size());
		std::transform(contents.begin(), contents.end(),
			std::back_inserter(vars),
			[](ade::TensptrT tens)
			{
				return std::make_shared<ead::VariableNode<PybindT>>(
					std::static_pointer_cast<ead::Variable<PybindT>>(tens));
			});
		eqns::VarErrsT varerrs = {
			{vars[0], grad_w},
			{vars[1], grad_hb},
			{vars[3], grad_vb},
		};

		updates_ = bbernoulli_approx(varerrs, learning_rate, discount_factor);

		ade::TensT to_track = {
			hidden_sample_->get_tensor(),
			visible_sample_->get_tensor(),
		};
		to_track.reserve(updates_.size() + 1);
		if (err_func)
		{
			error_ = err_func(ead::convert_to_node(visible_), visible_sample_);
			to_track.push_back(error_->get_tensor());
		}

		for (auto& assigns : updates_)
		{
			for (auto& assign : assigns)
			{
				auto source = assign.source_->get_tensor();
				assign_sources_.emplace(source.get());
				to_track.push_back(source);
			}
		}
		sess.track(to_track);
	}

	// Return error after training with train_in
	// if error is set, otherwise -1
	PybindT train (std::vector<PybindT>& train_in)
	{
		size_t insize = model_.get_ninput();
		if (train_in.size() != insize * batch_size_)
		{
			logs::fatalf("training vector size (%d) does not match "
				"input size (%d) * batchsize (%d)", train_in.size(),
				insize, batch_size_);
		}
		visible_->assign(train_in.data(), visible_->shape());

		sess_->update_target(assign_sources_, {
			visible_->get_tensor().get(),
		});

		if (nullptr == error_)
		{
			assign_groups(updates_,
				[this](ead::TensSetT& updated)
				{
					this->sess_->update(updated);
				});
			return -1;
		}

		assign_groups(updates_,
			[this](ead::TensSetT& updated)
			{
				this->sess_->update_target(
					ead::TensSetT{this->error_->get_tensor().get()}, updated);
			});
		return error_->data()[0];
	}

private:
	modl::RBM& model_;

	ead::VarptrT<PybindT> visible_ = nullptr;

	ead::NodeptrT<PybindT> hidden_sample_ = nullptr;

	ead::NodeptrT<PybindT> visible_sample_ = nullptr;

	ead::NodeptrT<PybindT> error_ = nullptr;

	// === updates && optimizer ===
	eqns::AssignGroupsT updates_;

	ead::TensSetT assign_sources_;

	ead::iSession* sess_;

	size_t batch_size_;
};

}

#endif // MODL_RBM_TRAINER_HPP
