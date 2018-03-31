//
// Created by Mingkai Chen on 2017-03-10.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzz.hpp"
#include "mock_subject.hpp"
#include "mock_observer.hpp"


#ifndef DISABLE_REACT_TEST


class REACT : public testutils::fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testutils::fuzz_test::TearDown();
		testify::mocker::clear();
	}
};


using namespace testutils;


// covers subject:
// clone, get_audience, attach, detach
TEST_F(REACT, CopySub_A000)
{
	mock_subject sassign1;
	mock_subject sassign2;

	mock_subject s1;
	mock_subject s2;
	mock_observer o1({&s2});
	
	std::vector<nnet::subject*> subjects = o1.expose_dependencies();
	ASSERT_EQ((size_t) 1, subjects.size());

	nnet::AUDMAP_T aud2 = s2.get_audience();
	EXPECT_EQ(0, s1.get_audience().size());
	EXPECT_EQ(1, aud2.size());
	EXPECT_TRUE(aud2.end() != aud2.find(&o1)) << 
		"expecting o1 to be an audience of s2";

	mock_subject cpy1(s1);
	mock_subject cpy2(s2);

	EXPECT_EQ(0, cpy1.get_audience().size());
	EXPECT_EQ(0, cpy2.get_audience().size());

	sassign1 = s1;
	sassign2 = s2;

	EXPECT_EQ(0, sassign1.get_audience().size());
	EXPECT_EQ(0, sassign2.get_audience().size());

	// detach to avoid doing anything with o1
	s1.detach(&o1);
	s2.detach(&o1);
	cpy1.detach(&o1);
	cpy2.detach(&o1);
	sassign2.detach(&o1);
	sassign2.detach(&o1);
}


// covers subject: move
TEST_F(REACT, MoveSub_A000)
{
	mock_subject sassign1;
	mock_subject sassign2;

	mock_subject s1;
	mock_subject s2;
	mock_observer o1({&s2});

	std::vector<nnet::subject*> subjects = o1.expose_dependencies();
	ASSERT_EQ((size_t) 1, subjects.size());

	nnet::AUDMAP_T aud2 = s2.get_audience();
	EXPECT_EQ(0, s1.get_audience().size());
	EXPECT_EQ(1, aud2.size());
	EXPECT_TRUE(aud2.end() != aud2.find(&o1)) << 
		"expecting o1 to be an audience of s2";

	mock_subject mv1(std::move(s1));
	mock_subject mv2(std::move(s2));

	size_t s1detach = testify::mocker::get_usage(&s1, "detach2");
	size_t s2detach = testify::mocker::get_usage(&s2, "detach2");
	EXPECT_EQ(0, s1detach);
	EXPECT_EQ(1, s2detach);

	aud2 = mv2.get_audience();
	EXPECT_EQ(0, s1.get_audience().size());
	EXPECT_EQ(0, s2.get_audience().size());
	EXPECT_EQ(0, mv1.get_audience().size());
	EXPECT_EQ(1, aud2.size());
	EXPECT_TRUE(aud2.end() != aud2.find(&o1)) << 
		"expecting o1 to be an audience of mv2";

	sassign1 = std::move(mv1);
	sassign2 = std::move(mv2);

	size_t mv1detach = testify::mocker::get_usage(&mv1, "detach2");
	size_t mv2detach = testify::mocker::get_usage(&mv2, "detach2");
	EXPECT_EQ(0, mv1detach);
	EXPECT_EQ(1, mv2detach);

	aud2 = sassign2.get_audience();
	EXPECT_EQ(0, mv1.get_audience().size());
	EXPECT_EQ(0, mv2.get_audience().size());
	EXPECT_EQ(0, sassign1.get_audience().size());
	EXPECT_EQ(1, aud2.size());
	EXPECT_TRUE(aud2.end() != aud2.find(&o1)) << 
		"expecting o1 to be an audience of sassign2";

	// detach to avoid doing anything with o1
	s1.detach(&o1);
	s2.detach(&o1);
	mv1.detach(&o1);
	mv2.detach(&o1);
	sassign2.detach(&o1);
	sassign2.detach(&o1);
}


// covers subject: notify, iobserver
TEST_F(REACT, Notify_A001)
{
	mock_subject s1;
	mock_subject s2;
	mock_observer o1({&s1, &s2});
	mock_observer o2({&s2});

	std::vector<nnet::subject*> subjects = o1.expose_dependencies();
	std::vector<nnet::subject*> subjects2 = o2.expose_dependencies();
	ASSERT_EQ((size_t) 2, subjects.size());
	ASSERT_EQ((size_t) 1, subjects2.size());

	s1.notify(nnet::UPDATE); // o1 update gets s1 at idx 0
	s2.notify(nnet::UPDATE);

	// o2 update gets s2 at idx 0,
	// o1 update gets s2 at idx 1
	size_t o1update = testify::mocker::get_usage(&o1, "update2");
	size_t o2update = testify::mocker::get_usage(&o2, "update2");
	optional<std::string> o1updateval = testify::mocker::get_value(&o1, "update2");
	optional<std::string> o2updateval = testify::mocker::get_value(&o2, "update2");
	ASSERT_TRUE((bool) o1updateval) <<
		"no label update2 for o1";
	ASSERT_TRUE((bool) o2updateval) <<
		"no label update2 for o2";
	ASSERT_EQ(2, o1update);
	ASSERT_EQ(1, o2update);
	EXPECT_STREQ("UPDATE", o1updateval->c_str());
	EXPECT_STREQ("UPDATE", o2updateval->c_str());

	// suicide calls
	s1.notify(nnet::UNSUBSCRIBE);
	s2.notify(nnet::UNSUBSCRIBE);

	o1update = testify::mocker::get_usage(&o1, "update2");
	o2update = testify::mocker::get_usage(&o2, "update2");
	o1updateval = testify::mocker::get_value(&o1, "update2");
	o2updateval = testify::mocker::get_value(&o2, "update2");
	ASSERT_TRUE((bool) o1updateval) <<
		"no label update2 for o1";
	ASSERT_TRUE((bool) o2updateval) <<
		"no label update2 for o2";
	ASSERT_EQ(4, o1update);
	ASSERT_EQ(2, o2update);
	EXPECT_STREQ("UNSUBSCRIBE", o1updateval->c_str());
	EXPECT_STREQ("UNSUBSCRIBE", o2updateval->c_str());

	// detach to avoid doing anything with o1 and o2
	s1.detach(&o1);
	s2.detach(&o1);
	s2.detach(&o2);
}


// covers subject: destructor
TEST_F(REACT, SUBDEATH_A002)
{
	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer* o1 = new mock_observer({s1, s2});
	mock_observer* o2 = new mock_observer({s2});

	std::vector<nnet::subject*> subjects = o1->expose_dependencies();
	std::vector<nnet::subject*> subjects2 = o2->expose_dependencies();
	ASSERT_EQ((size_t) 2, subjects.size());
	ASSERT_EQ((size_t) 1, subjects2.size());

	// suicide calls
	delete s1;
	size_t o1update = testify::mocker::get_usage(o1, "update2");
	optional<std::string> o1updateval = testify::mocker::get_value(o1, "update2");
	ASSERT_TRUE((bool) o1updateval) <<
		"no label update2 for o1";
	ASSERT_EQ(1, o1update);
	EXPECT_STREQ("UNSUBSCRIBE", o1updateval->c_str());
	o1->set_label("update2", "");

	delete s2;
	o1update = testify::mocker::get_usage(o1, "update2");
	size_t o2update = testify::mocker::get_usage(o2, "update2");
	o1updateval = testify::mocker::get_value(o1, "update2");
	optional<std::string> o2updateval = testify::mocker::get_value(o2, "update2");
	ASSERT_TRUE((bool) o1updateval) <<
		"no label update2 for o1";
	ASSERT_TRUE((bool) o2updateval) <<
		"no label update2 for o2";
	ASSERT_EQ(2, o1update);
	ASSERT_EQ(1, o2update);
	EXPECT_STREQ("UNSUBSCRIBE", o1updateval->c_str());
	EXPECT_STREQ("UNSUBSCRIBE", o2updateval->c_str());

	delete o1;
	delete o2;
}


// covers subject: attach
TEST_F(REACT, Attach_A003)
{
	mock_observer o1;
	mock_observer o2;
	mock_subject s1;
	mock_subject s2;

	EXPECT_EQ(0, s1.get_audience().size());
	EXPECT_EQ(0, s2.get_audience().size());

	size_t i = get_int(1, "i", {0, 102})[0];
	s1.attach(&o1, i);
	s2.attach(&o2, i);
	s2.attach(&o1, i + 1);

	EXPECT_EQ(1, s1.get_audience().size());
	EXPECT_EQ(2, s2.get_audience().size());

	s1.attach(&o1, i + 1);
	s1.detach(&o1);
	s2.detach(&o2);

	EXPECT_EQ(0, s1.get_audience().size());
	EXPECT_EQ(1, s2.get_audience().size());

	s1.detach(&o1);
	s2.detach(&o1);
	s2.detach(&o2);
}


// covers subject: detach without index and with index
TEST_F(REACT, Detach_A004)
{
	mock_subject s1;
	mock_subject s2;
	mock_subject s3;
	mock_observer o1({&s1});
	mock_observer o2({&s1, &s2, &s2});
	mock_observer o3({&s3, &s3});

	EXPECT_EQ(2, s1.get_audience().size());
	s1.detach(&o1);
	EXPECT_EQ(1, s1.get_audience().size());
	s1.detach(&o2);
	EXPECT_EQ(0, s1.get_audience().size());

	EXPECT_EQ(1, s2.get_audience().size());
	s2.detach(&o2);
	EXPECT_EQ(0, s2.get_audience().size());

	EXPECT_EQ(1, s3.get_audience().size());
	s3.detach(&o3, 1);
	EXPECT_EQ(1, s3.get_audience().size());
	s3.detach(&o3, 0);
	EXPECT_EQ(0, s3.get_audience().size());

	o1.mock_clear_dependency();
	o2.mock_clear_dependency();
	o3.mock_clear_dependency();
}


// covers iobserver
// default and dependency constructors
TEST_F(REACT, ObsConstruct_A005)
{
	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer* o1 = new mock_observer({s2});
	mock_observer* o2 = new mock_observer({s1, s2});

	std::vector<nnet::subject*> subs1 = o1->expose_dependencies();
	std::vector<nnet::subject*> subs2 = o2->expose_dependencies();

	ASSERT_EQ((size_t) 1, subs1.size());
	EXPECT_EQ(s2, subs1[0]);
	ASSERT_EQ((size_t) 2, subs2.size());
	EXPECT_EQ(s1, subs2[0]);
	EXPECT_EQ(s2, subs2[1]);

	// called twice since mock observer isn't destroyed when death_on_broken is called
	// so deleting s2 will trigger another suicide call
	delete s1;
	delete s2;

	size_t o1death = testify::mocker::get_usage(o1, "death_on_broken");
	size_t o2death = testify::mocker::get_usage(o2, "death_on_broken");
	EXPECT_EQ(1, o1death);
	EXPECT_EQ(2, o2death);

	// again observers aren't destroyed
	delete o1;
	delete o2;
}


// covers iobserver: clone
TEST_F(REACT, CopyObs_A006)
{
	mock_observer* sassign1 = new mock_observer;
	mock_observer* sassign2 = new mock_observer;

	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer* o1 = new mock_observer({s2});
	mock_observer* o2 = new mock_observer({s1, s2});

	mock_observer* cpy1 = new mock_observer(*o1);
	mock_observer* cpy2 = new mock_observer(*o2);

	std::vector<nnet::subject*> subs1 = cpy1->expose_dependencies();
	std::vector<nnet::subject*> subs2 = cpy2->expose_dependencies();
	ASSERT_EQ((size_t) 1, subs1.size());
	EXPECT_EQ(s2, subs1[0]);
	ASSERT_EQ((size_t) 2, subs2.size());
	EXPECT_EQ(s1, subs2[0]);
	EXPECT_EQ(s2, subs2[1]);

	*sassign1 = *o1;
	*sassign2 = *o2;

	std::vector<nnet::subject*> subs3 = sassign1->expose_dependencies();
	std::vector<nnet::subject*> subs4 = sassign2->expose_dependencies();
	ASSERT_EQ((size_t) 1, subs3.size());
	EXPECT_EQ(s2, subs3[0]);
	ASSERT_EQ((size_t) 2, subs4.size());
	EXPECT_EQ(s1, subs4[0]);
	EXPECT_EQ(s2, subs4[1]);

	delete s1;
	size_t o1death = testify::mocker::get_usage(o1, "death_on_broken");
	size_t o2death = testify::mocker::get_usage(o2, "death_on_broken");
	size_t cpy1death = testify::mocker::get_usage(cpy1, "death_on_broken");
	size_t cpy2death = testify::mocker::get_usage(cpy2, "death_on_broken");
	size_t sassign1death = testify::mocker::get_usage(sassign1, "death_on_broken");
	size_t sassign2death = testify::mocker::get_usage(sassign2, "death_on_broken");
	EXPECT_EQ(0, o1death);
	EXPECT_EQ(1, o2death);
	EXPECT_EQ(0, cpy1death);
	EXPECT_EQ(1, cpy2death);
	EXPECT_EQ(0, sassign1death);
	EXPECT_EQ(1, sassign2death);

	delete s2;
	o1death = testify::mocker::get_usage(o1, "death_on_broken");
	o2death = testify::mocker::get_usage(o2, "death_on_broken");
	cpy1death = testify::mocker::get_usage(cpy1, "death_on_broken");
	cpy2death = testify::mocker::get_usage(cpy2, "death_on_broken");
	sassign1death = testify::mocker::get_usage(sassign1, "death_on_broken");
	sassign2death = testify::mocker::get_usage(sassign2, "death_on_broken");
	EXPECT_EQ(1, o1death);
	EXPECT_EQ(2, o2death);
	EXPECT_EQ(1, cpy1death);
	EXPECT_EQ(2, cpy2death);
	EXPECT_EQ(1, sassign1death);
	EXPECT_EQ(2, sassign2death);

	delete o1;
	delete o2;
	delete cpy1;
	delete cpy2;
	delete sassign1;
	delete sassign2;
}


// covers iobserver: move
TEST_F(REACT, MoveObs_A006)
{
	mock_observer* sassign1 = new mock_observer;
	mock_observer* sassign2 = new mock_observer;

	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer* o1 = new mock_observer({s2});
	mock_observer* o2 = new mock_observer({s1, s2});

	mock_observer* mv1 = new mock_observer(std::move(*o1));
	mock_observer* mv2 = new mock_observer(std::move(*o2));

	std::vector<nnet::subject*> subs1 = mv1->expose_dependencies();
	std::vector<nnet::subject*> subs2 = mv2->expose_dependencies();
	ASSERT_EQ((size_t) 1, subs1.size());
	EXPECT_EQ(s2, subs1[0]);
	ASSERT_EQ((size_t) 2, subs2.size());
	EXPECT_EQ(s1, subs2[0]);
	EXPECT_EQ(s2, subs2[1]);

	EXPECT_EQ(0, o1->expose_dependencies().size());
	EXPECT_EQ(0, o2->expose_dependencies().size());

	*sassign1 = std::move(*mv1);
	*sassign2 = std::move(*mv2);

	std::vector<nnet::subject*> subs3 = sassign1->expose_dependencies();
	std::vector<nnet::subject*> subs4 = sassign2->expose_dependencies();
	ASSERT_EQ((size_t) 1, subs3.size());
	EXPECT_EQ(s2, subs3[0]);
	ASSERT_EQ((size_t) 2, subs4.size());
	EXPECT_EQ(s1, subs4[0]);
	EXPECT_EQ(s2, subs4[1]);

	EXPECT_EQ(0, mv1->expose_dependencies().size());
	EXPECT_EQ(0, mv2->expose_dependencies().size());

	delete s1;
	size_t o1death = testify::mocker::get_usage(o1, "death_on_broken");
	size_t o2death = testify::mocker::get_usage(o2, "death_on_broken");
	size_t mv1death = testify::mocker::get_usage(mv1, "death_on_broken");
	size_t mv2death = testify::mocker::get_usage(mv2, "death_on_broken");
	size_t sassign1death = testify::mocker::get_usage(sassign1, "death_on_broken");
	size_t sassign2death = testify::mocker::get_usage(sassign2, "death_on_broken");
	EXPECT_EQ(0, o1death);
	EXPECT_EQ(0, o2death);
	EXPECT_EQ(0, mv1death);
	EXPECT_EQ(0, mv2death);
	EXPECT_EQ(0, sassign1death);
	EXPECT_EQ(1, sassign2death);

	delete s2;
	o1death = testify::mocker::get_usage(o1, "death_on_broken");
	o2death = testify::mocker::get_usage(o2, "death_on_broken");
	mv1death = testify::mocker::get_usage(mv1, "death_on_broken");
	mv2death = testify::mocker::get_usage(mv2, "death_on_broken");
	sassign1death = testify::mocker::get_usage(sassign1, "death_on_broken");
	sassign2death = testify::mocker::get_usage(sassign2, "death_on_broken");
	EXPECT_EQ(0, o1death);
	EXPECT_EQ(0, o2death);
	EXPECT_EQ(0, mv1death);
	EXPECT_EQ(0, mv2death);
	EXPECT_EQ(1, sassign1death);
	EXPECT_EQ(2, sassign2death);

	delete o1;
	delete o2;
	delete mv1;
	delete mv2;
	delete sassign1;
	delete sassign2;
}


// covers iobserver: add_dependency
TEST_F(REACT, AddDep_A007)
{
	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer* o1 = new mock_observer;
	mock_observer* o2 = new mock_observer;

	nnet::AUDMAP_T aud1 = s1->get_audience();
	nnet::AUDMAP_T aud2 = s2->get_audience();
	EXPECT_EQ(0, aud1.size());
	EXPECT_EQ(0, aud2.size());
	o1->mock_add_dependency(s2);
	aud2 = s2->get_audience();
	EXPECT_EQ(1, aud2.size());
	EXPECT_TRUE(aud2.end() != aud2.find(o1)) <<
		"o1 not observing s2";
	o1->mock_add_dependency(s1);
	aud1 = s1->get_audience();
	EXPECT_EQ(1, aud1.size());
	EXPECT_TRUE(aud1.end() != aud1.find(o1)) <<
		"o1 not observing s1";

	o2->mock_add_dependency(s1);
	o2->mock_add_dependency(s2);

	std::vector<nnet::subject*> subs1 = o1->expose_dependencies();
	std::vector<nnet::subject*> subs2 = o2->expose_dependencies();

	ASSERT_EQ((size_t) 2, subs1.size());
	ASSERT_EQ((size_t) 2, subs2.size());

	EXPECT_EQ(s2, subs1[0]);
	EXPECT_EQ(s1, subs1[1]);
	EXPECT_EQ(s1, subs2[0]);
	EXPECT_EQ(s2, subs2[1]);

	delete s1;
	delete s2;
	size_t o1death = testify::mocker::get_usage(o1, "death_on_broken");
	size_t o2death = testify::mocker::get_usage(o2, "death_on_broken");
	EXPECT_EQ(2, o1death);
	EXPECT_EQ(2, o2death);
	delete o1;
	delete o2;
}


// covers iobserver: remove_dependency
TEST_F(REACT, RemDep_A008)
{
	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer* o1 = new mock_observer({s2});
	mock_observer* o2 = new mock_observer({s1, s2});
	mock_observer* o3 = new mock_observer({s1, s2});

	// out of index removal
	o1->mock_remove_dependency(1);
	EXPECT_EQ(1, o1->expose_dependencies().size());
	o2->mock_remove_dependency(2);
	EXPECT_EQ(2, o2->expose_dependencies().size());
	o3->mock_remove_dependency(2);
	EXPECT_EQ(2, o3->expose_dependencies().size());

	// proper removal
	o1->mock_remove_dependency(0);
	EXPECT_EQ(0, o1->expose_dependencies().size());

	o2->mock_remove_dependency(1);
	EXPECT_EQ(1, o2->expose_dependencies().size());
	o2->mock_remove_dependency(0);
	EXPECT_EQ(0, o2->expose_dependencies().size());

	o3->mock_remove_dependency(0);
	EXPECT_EQ(2, o3->expose_dependencies().size());
	o3->mock_remove_dependency(1);
	EXPECT_EQ(0, o3->expose_dependencies().size());

	delete s1;
	delete s2;
	delete o1;
	delete o2;
	delete o3;
}


// covers iobserver: replace_dependency
TEST_F(REACT, RepDep_A009)
{
	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer* o1 = new mock_observer({s1});

	o1->mock_replace_dependency(s1, 1);
	ASSERT_EQ(1, o1->expose_dependencies().size());
	o1->mock_replace_dependency(s2, 0);
	std::vector<nnet::subject*> subs1 = o1->expose_dependencies();
	ASSERT_EQ(1, subs1.size());
	EXPECT_EQ(s2, subs1[0]);

	delete s1;
	delete s2;
	EXPECT_EQ(1, testify::mocker::get_usage(o1, "death_on_broken"));
	delete o1;
}


// covers iobserver
// destruction, depends on subject detach
TEST_F(REACT, ObsDeath_A010)
{
	mock_subject s1;
	mock_subject s2;

	mock_observer* o1 = new mock_observer({&s1, &s2});
	nnet::iobserver* tempptr = o1;
	delete o1;
	EXPECT_EQ(1, testify::mocker::get_usage(&s1, "detach2"));
	EXPECT_EQ(0, s2.get_audience().size());
	s1.detach(tempptr);
}


#endif /* DISABLE_REACT_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
