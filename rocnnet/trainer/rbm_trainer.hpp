#include "layr/rbm.hpp"

#include "layr/err_approx.hpp"

#ifndef RCN_RBM_TRAINER_HPP
#define RCN_RBM_TRAINER_HPP

namespace trainer
{

// Bernoulli RBM "error approximation"
// for each (x, err) in leaves
// momentum_next ~ χ * momentum_cur + η * (1 - χ) / err.shape[0] * err
// x_next = x_curr + next_momentum
//
// where η is the learning rate, and χ is discount_factor
layr::AssignGroupsT bbernoulli_approx (const layr::VarErrsT& leaves,
	PybindT learning_rate, PybindT discount_factor,
	std::string root_label = "")
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

using ErrorF = std::function<eteq::NodeptrT<PybindT>(eteq::NodeptrT<PybindT>,eteq::NodeptrT<PybindT>)>;

struct BernoulliRBMTrainer final
{
	BernoulliRBMTrainer (layr::RBM& model,
		eteq::iSession& sess,
		teq::DimT batch_size,
		PybindT learning_rate,
		PybindT discount_factor,
		ErrorF err_func = ErrorF()) :
		model_(model), sess_(&sess), batch_size_(batch_size)
	{
		visible_ = eteq::make_variable_scalar<PybindT>(0,
			teq::Shape({(teq::DimT) model.get_ninput(), batch_size}));

		hidden_sample_ = model.connect(visible_);
		visible_sample_ = model.backward_connect(
			tenncor::random::rand_binom_one(hidden_sample_));

		auto hidden_reconp = model.connect(visible_sample_);

		auto grad_w =
			tenncor::matmul(tenncor::transpose(
				eteq::convert_to_node(visible_)), hidden_sample_) -
			tenncor::matmul(tenncor::transpose(
				visible_sample_), hidden_reconp);
		auto grad_hb = tenncor::reduce_mean_1d(
			hidden_sample_ - hidden_reconp, 1);
		auto grad_vb = tenncor::reduce_mean_1d(
			eteq::convert_to_node(visible_) - visible_sample_, 1);

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

		updates_ = bbernoulli_approx(varerrs, learning_rate, discount_factor);

		teq::TensT to_track = {
			hidden_sample_->get_tensor(),
			visible_sample_->get_tensor(),
		};
		to_track.reserve(updates_.size() + 1);
		if (err_func)
		{
			error_ = err_func(eteq::convert_to_node(visible_), visible_sample_);
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
				[this](eteq::TensSetT& updated)
				{
					this->sess_->update(updated);
				});
			return -1;
		}

		assign_groups(updates_,
			[this](eteq::TensSetT& updated)
			{
				this->sess_->update_target(
					eteq::TensSetT{this->error_->get_tensor().get()}, updated);
			});
		return error_->data()[0];
	}

private:
	layr::RBM& model_;

	eteq::VarptrT<PybindT> visible_ = nullptr;

	eteq::NodeptrT<PybindT> hidden_sample_ = nullptr;

	eteq::NodeptrT<PybindT> visible_sample_ = nullptr;

	eteq::NodeptrT<PybindT> error_ = nullptr;

	// === updates && optimizer ===
	layr::AssignGroupsT updates_;

	eteq::TensSetT assign_sources_;

	eteq::iSession* sess_;

	size_t batch_size_;
};

}

#endif // RCN_RBM_TRAINER_HPP
