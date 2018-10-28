#include "rocnnet/modl/mlp.hpp"

#include "rocnnet/eqns/err_approx.hpp"

// GDTrainer does not own anything
struct GDTrainer
{
	GDTrainer (MLP& brain, ApproxFuncT update,
		uint8_t batch_size, std::string label) :
		label_(label), brain_(&brain), batch_size_(batch_size),
		train_in_(ade::Shape({brain.get_ninput(), batch_size})),
		expected_out_(ade::Shape({brain.get_noutput(), batch_size})),
		// todo: move error out of initializer list to avoid confusing order of init
		error_(llo::pow(llo::sub(expected_out_, brain(train_in_)),
			llo::shaped_scalar<double>(2, expected_out_.tensor_->shape())))
	{
		updates_ = update(error_, brain.get_variables());
	}

	void train (std::vector<double>& train_in,
		std::vector<double>& expected_out)
	{
		size_t insize = brain_->get_ninput();
		size_t outsize = brain_->get_noutput();
		if (train_in.size() != insize * batch_size_)
		{
			ade::fatalf("training vector size (%d) does not match input size "
				"(%d) * batchsize (%d)", train_in.size(), insize, batch_size_);
		}
		if (expected_out.size() != outsize * batch_size_)
		{
			ade::fatalf("expected output size (%d) does not match output size "
				"(%d) * batchsize (%d)", expected_out.size(), outsize, batch_size_);
		}
		train_in_ = train_in;

		expected_out_ = expected_out;
		std::unordered_map<llo::iSource*,llo::GenericData> data;
		for (auto& varpair : updates_)
		{
			data[varpair.first] = varpair.second.data(llo::DOUBLE);
		}
		for (auto& datapair : data)
		{
			datapair.first->reassign(datapair.second);
		}
	}

	std::string label_;
	MLP* brain_ = nullptr; // do not own this
	uint8_t batch_size_;
	llo::PlaceHolder<double> train_in_;
	llo::PlaceHolder<double> expected_out_;
	llo::DataNode error_;

	DeltasT updates_;
};
