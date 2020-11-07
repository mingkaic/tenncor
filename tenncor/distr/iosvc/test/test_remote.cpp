
#ifndef DISABLE_IOSVC_REMOTE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"
#include "tenncor/distr/mock/mock.hpp"
#include "tenncor/distr/iosvc/mock/mock.hpp"

#include "tenncor/distr/iosvc/iosvc.hpp"


const std::string test_service = "tenncor.distr.iosvc.test";


struct REMOTE : public ::testing::Test, public DistrTestcase
{
protected:
	void TearDown (void) override
	{
		MockServerBuilder::clear_service();
	}

	distr::iDistrMgrptrT make_mgr (const std::string& id)
	{
		return make_mgr(id, reserve_port());
	}

	distr::iDistrMgrptrT make_mgr (const std::string& id, size_t port)
	{
		return DistrTestcase::make_local_mgr(port, {
			register_mock_iosvc,
		}, id);
	}
};


TEST_F(REMOTE, LocalReferenceStorage)
{
	distr::iDistrMgrptrT mgr(make_mgr("mgr"));
	auto& service = distr::get_iosvc(*mgr);

	teq::Shape outshape({2, 2});
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	service.expose_node(a);

	// shouldn't find references if we only exposed a from local
	EXPECT_EQ(0, service.get_remotes().size());

	auto ida = *service.lookup_id(a.get());
	error::ErrptrT err = nullptr;
	service.lookup_node(err, ida);

	// shouldn't find references
	EXPECT_EQ(0, service.get_remotes().size());
}


TEST_F(REMOTE, RemoteReferenceStorage)
{
	distr::iDistrMgrptrT mgr(make_mgr("mgr"));
	auto& service = distr::get_iosvc(*mgr);

	teq::Shape outshape({2, 2});
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	a->meta_.tcode_ = egen::DOUBLE;
	a->meta_.tname_ = "DOUBLE";
	service.expose_node(a);

	distr::iDistrMgrptrT mgr2(make_mgr("mgr2"));
	auto& service2 = distr::get_iosvc(*mgr2);

	EXPECT_EQ(0, service.get_remotes().size());

	auto ida = *service.lookup_id(a.get());
	error::ErrptrT err = nullptr;
	service2.lookup_node(err, ida);

	ASSERT_NOERR(err);
	EXPECT_EQ(1, service2.get_remotes().size());
}


#endif // DISABLE_IOSVC_REMOTE_TEST
