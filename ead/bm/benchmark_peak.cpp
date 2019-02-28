#include <random>
#include <chrono>

#include "ade/shape.hpp"

#include "ead/eigen.hpp"


static std::random_device rnd_device;
static std::mt19937 mersenne_engine(rnd_device());


static std::vector<double> random_data (size_t n, double lower, double upper)
{
	std::vector<double> out(n);
	std::uniform_real_distribution<double> dist(lower, upper);
	std::generate(out.begin(), out.end(),
		[&dist]() { return dist(mersenne_engine); });
	return out;
}


int main (int argc, char** argv)
{
	ade::Shape in_shape({10, 3});
	ade::Shape weight0_shape({9, 10});
	ade::Shape bias0_shape({9});
	ade::Shape weight1_shape({5, 9});
	ade::Shape bias1_shape({5});
	ade::Shape out_shape({5,3});

	Eigen::Matrix<double,3,10,Eigen::RowMajor> in;
	Eigen::Matrix<double,10,9,Eigen::RowMajor> weight0;
	Eigen::Matrix<double,1,9> bias0;
	Eigen::Matrix<double,9,5,Eigen::RowMajor> weight1;
	Eigen::Matrix<double,1,5> bias1;
	Eigen::Matrix<double,3,5,Eigen::RowMajor> out;

	auto layer0 = (in * weight0).rowwise() + bias0;
	auto sig0 = layer0.unaryExpr(Eigen::internal::scalar_sigmoid_op<double>());

	auto layer1 = (sig0 * weight1).rowwise() + bias1;
	auto sig1 = layer1.unaryExpr(Eigen::internal::scalar_sigmoid_op<double>());

	Eigen::TensorMap<Eigen::Tensor<double,3>> intens(in.data(), 10, 3, 1);
	Eigen::TensorMap<Eigen::Tensor<double,3>> w1tens(weight1.data(), 5, 9, 1);

	Eigen::Matrix<double,3,9,Eigen::RowMajor> sig0_in;
	Eigen::Matrix<double,3,5,Eigen::RowMajor> two_four_in;
	Eigen::TensorMap<Eigen::Tensor<double,2>> sig0ten(sig0_in.data(), 9, 3);
	Eigen::TensorMap<Eigen::Tensor<double,3>> two_fourten(two_four_in.data(), 5, 3, 1);

	auto two_four = sig1.unaryExpr([](double d) { return (1.0 - d) * d; })
		.cwiseProduct(out - sig1) * -2;

	// sig0ten has shape = [9, 3, 1]
	// broadcast -> [9, 3, 5]
	// shuffle -> [5, 3, 9]
	auto four_three = sig0ten.reshape(std::array<Eigen::Index,3>{9, 3, 1})
		.broadcast(std::array<Eigen::Index,3>{1, 1, 5})
		.shuffle(std::array<Eigen::Index,3>{2, 1, 0});

	// two_fourten has shape = [5, 3, 1]
	// broadcast -> [5, 3, 9]
	auto two_three = two_fourten.broadcast(std::array<Eigen::Index,3>{1, 1, 9});

	auto seven = (w1tens.broadcast(std::array<Eigen::Index,3>{1, 1, 3})
		.shuffle(std::array<Eigen::Index,3>{0, 2, 1}) * two_three)
		.shuffle(std::array<Eigen::Index,3>{2, 1, 0})
		.sum(std::array<Eigen::Index,1>{2}) * (1 - sig0ten) * sig0ten;

	auto dw0 = (intens.broadcast(std::array<Eigen::Index,3>{1, 1, 9})
		.shuffle(std::array<Eigen::Index,3>{2, 1, 0}) *
		seven.reshape(std::array<Eigen::Index,3>{9, 3, 1})
		.broadcast(std::array<Eigen::Index,3>{1, 1, 10}))
		.shuffle(std::array<Eigen::Index,3>{0, 2, 1})
		.sum(std::array<Eigen::Index,1>{2});
	auto db0 = seven
		.sum(std::array<Eigen::Index,1>{1});
	auto dw1 = (four_three * two_three)
		.sum(std::array<Eigen::Index,1>{1});


	Eigen::Tensor<double,2> dw0_in(9, 10);
	Eigen::Tensor<double,1> db0_in(9);
	Eigen::Tensor<double,2> dw1_in(5, 9);
	Eigen::Map<Eigen::Matrix<double,10,9,Eigen::RowMajor>> dw0_mat(dw0_in.data(), 10, 9);
	Eigen::Map<Eigen::Matrix<double,1,9,Eigen::RowMajor>> db0_mat(db0_in.data(), 1, 9);
	Eigen::Map<Eigen::Matrix<double,9,5,Eigen::RowMajor>> dw1_mat(dw1_in.data(), 9, 5);
	auto db1_mat = two_four.colwise().sum();

	double learning_rate = 0.9;

	auto dw0_error = dw0_mat * learning_rate;
	auto db0_error = db0_mat * learning_rate;
	auto dw1_error = dw1_mat * learning_rate;
	auto db1_error = db1_mat * learning_rate;

	std::vector<double> in_data = random_data(in_shape.n_elems(), 0, 1);
	std::vector<double> w0_data = random_data(weight0_shape.n_elems(), 0, 1);
	std::vector<double> b0_data = random_data(bias0_shape.n_elems(), 0, 1);
	std::vector<double> w1_data = random_data(weight1_shape.n_elems(), 0, 1);
	std::vector<double> b1_data = random_data(bias1_shape.n_elems(), 0, 1);
	std::vector<double> out_data = random_data(out_shape.n_elems(), 0, 1);
	std::copy(in.data(), in.data() + in_shape.n_elems(), in_data.data());
	std::copy(weight0.data(), weight0.data() + weight0_shape.n_elems(), w0_data.data());
	std::copy(bias0.data(), bias0.data() + bias0_shape.n_elems(), b0_data.data());
	std::copy(weight1.data(), weight1.data() + weight1_shape.n_elems(), w1_data.data());
	std::copy(bias1.data(), bias1.data() + bias1_shape.n_elems(), b1_data.data());
	std::copy(out.data(), out.data() + out_shape.n_elems(), out_data.data());

	// todo: check correctness of this equation
	auto start = std::chrono::steady_clock::now();

	sig0_in = sig0;
	two_four_in = two_four;

	dw0_in = dw0;
	db0_in = db0;
	dw1_in = dw1;

	weight0 -= dw0_error;
	bias0 -= db0_error;
	weight1 -= dw1_error;
	bias1 -= db1_error;

	auto end = std::chrono::steady_clock::now();
	std::cout << "duration: " << std::chrono::duration_cast<
		std::chrono::nanoseconds>(end - start).count() << "ns\n";

	return 0;
}
