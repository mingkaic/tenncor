
#ifndef DISABLE_GLOBAL_LOG_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "exam/exam.hpp"

#include "internal/global/global.hpp"


TEST(LOGGER, SetGet)
{
	auto levelstr = exam::TestLogger::get_llevel =
		"SetGet" + fmts::to_string(std::this_thread::get_id());

	EXPECT_NE(nullptr, dynamic_cast<global::G3Logger*>(&global::get_logger()));
	global::set_logger(new exam::TestLogger());
	auto logger = dynamic_cast<exam::TestLogger*>(&global::get_logger());
	ASSERT_NE(nullptr, logger);

	global::infof("hello %s", "world");
	EXPECT_EQ(logs::INFO, logger->latest_lvl_);
	EXPECT_STREQ("hello world", logger->latest_msg_.c_str());
	global::debugf("hello %d", 0);
	EXPECT_EQ(logs::DEBUG, logger->latest_lvl_);
	EXPECT_STREQ("hello 0", logger->latest_msg_.c_str());
	global::warnf("hello %d", 42);
	EXPECT_EQ(logs::WARN, logger->latest_lvl_);
	EXPECT_STREQ("hello 42", logger->latest_msg_.c_str());
	global::errorf("hello %f", 0.0);
	EXPECT_EQ(logs::ERROR, logger->latest_lvl_);
	EXPECT_STREQ("hello 0.000000", logger->latest_msg_.c_str());
	EXPECT_FATAL(global::throw_errf("hello %f", 1.0), "hello 1.000000");
	EXPECT_FATAL(global::fatalf("hello %f", 2.0), "hello 2.000000");

	global::set_log_level(logs::fatal_level);
	EXPECT_STREQ(logs::fatal_level.c_str(), exam::TestLogger::set_llevel.c_str());
	EXPECT_STREQ(levelstr.c_str(), global::get_log_level().c_str());

	global::set_logger(nullptr);
	EXPECT_EQ(nullptr, dynamic_cast<exam::TestLogger*>(&global::get_logger()));
	EXPECT_NE(nullptr, dynamic_cast<global::G3Logger*>(&global::get_logger()));
}


TEST(LOGGER, FmtLogger)
{
	auto levelstr = exam::TestLogger::get_llevel =
		"FmtLogger" + fmts::to_string(std::this_thread::get_id());

	exam::TestLogger baselogger;
	global::FormatLogger fmter(baselogger, "abcd ", " efgh");

	EXPECT_FALSE(fmter.supports_level("not a real level"));
	EXPECT_TRUE(fmter.supports_level(logs::fatal_level));

	EXPECT_FALSE(fmter.supports_level(logs::NOT_SET));
	EXPECT_TRUE(fmter.supports_level(logs::FATAL));

	fmter.log(logs::error_level, "zzzxxxyyy");
	EXPECT_EQ(logs::ERROR, exam::TestLogger::latest_lvl_);
	EXPECT_STREQ("abcd zzzxxxyyy efgh", exam::TestLogger::latest_msg_.c_str());

	fmter.log(logs::WARN, "kkjluvip");
	EXPECT_EQ(logs::WARN, exam::TestLogger::latest_lvl_);
	EXPECT_STREQ("abcd kkjluvip efgh", exam::TestLogger::latest_msg_.c_str());

	fmter.set_log_level("abcd");
	EXPECT_STREQ("abcd", exam::TestLogger::set_llevel.c_str());
	EXPECT_STREQ(levelstr.c_str(), fmter.get_log_level().c_str());

	exam::TestLogger::set_llevel = "";
}


struct TestSink
{
	TestSink (std::unordered_map<size_t,std::string>* msgs) :
		msgs_(msgs) {}

	void ReceiveLogMessage (g3::LogMessageMover entry)
	{
		auto level = entry.get()._level.value;
		auto msg = entry.get().toString(
			[](const g3::LogMessage& msg) -> std::string
			{ return "g3 message "; });
		(*msgs_)[level] = msg;
	}

	std::unordered_map<size_t,std::string>* msgs_;
};


TEST(LOGGER, G3Logger)
{
	global::set_logger(new exam::TestLogger());
	std::unordered_map<size_t,std::string> msgs;
	std::stringstream outs;
	{
		global::G3Logger log(
			[&](global::G3WorkerT& worker)
			{
				global::add_stdio_sink(worker, outs);
				worker->addSink(std::make_unique<TestSink>(&msgs),
					&TestSink::ReceiveLogMessage);
			});
		log.set_log_level(logs::debug_level);
		EXPECT_STREQ(logs::debug_level.c_str(), log.get_log_level().c_str());

		EXPECT_FALSE(log.supports_level("not a real level"));
		EXPECT_TRUE(log.supports_level(logs::fatal_level));

		EXPECT_FALSE(log.supports_level(std::numeric_limits<size_t>::max()));
		EXPECT_TRUE(log.supports_level(FATAL.value));

		ASSERT_TRUE(log.supports_level(logs::debug_level));
		log.log(logs::debug_level, "aayyy boi");

		ASSERT_TRUE(log.supports_level(logs::info_level));
		log.log(logs::info_level, "zzzxxxyyy");

		ASSERT_TRUE(log.supports_level(WARNING.value));
		log.log(WARNING.value, "kkjluvip");

		ASSERT_TRUE(log.supports_level(ERROR.value));
		log.log(logs::error_level, "zzzerrormsg");

		try
		{
			log.log(logs::throw_err_level, "of huck");
		}
		catch (std::runtime_error& e)
		{
			EXPECT_STREQ("of huck", e.what());
		}
		catch (std::exception& e)
		{
			FAIL() << "unexpected throw " << e.what();
		}

		log.set_log_level("abcd");
		EXPECT_STREQ(logs::debug_level.c_str(), log.get_log_level().c_str());

		log.set_log_level(FATAL.text);
		EXPECT_STREQ(logs::fatal_level.c_str(), log.get_log_level().c_str());

		log.set_log_level(logs::throw_err_level);
		EXPECT_STREQ(logs::throw_err_level.c_str(), log.get_log_level().c_str());
	}
	global::set_logger(nullptr);
	// delay message evaluation until initial g3 log has shutdown and reset
	// (this assumes all previous entries have been processed since shutdown should gracefully wait)
	ASSERT_HAS(msgs, DBUG.value);
	ASSERT_HAS(msgs, INFO.value);
	ASSERT_HAS(msgs, WARNING.value);
	ASSERT_HAS(msgs, ERROR.value);
	ASSERT_HAS(msgs, THROW_ERR.value);
	EXPECT_STREQ("g3 message aayyy boi\n", msgs[DBUG.value].c_str());
	EXPECT_STREQ("g3 message zzzxxxyyy\n", msgs[INFO.value].c_str());
	EXPECT_STREQ("g3 message kkjluvip\n", msgs[WARNING.value].c_str());
	EXPECT_STREQ("g3 message zzzerrormsg\n", msgs[ERROR.value].c_str());
	EXPECT_STREQ("g3 message of huck\n", msgs[THROW_ERR.value].c_str());

	auto lines = fmts::split(outs.str(), "\n");
	ASSERT_EQ(6, lines.size());
	types::StringsT messages;
	for (size_t i = 0; i < 5; ++i)
	{
		auto line = lines[i];
		auto seps = fmts::split(line, "\t");
		auto msg = fmts::join("\t", seps.begin() + 1, seps.end());
		messages.push_back(msg);
	}
	std::sort(messages.begin(), messages.end());
	EXPECT_STREQ("aayyy boi", messages[0].c_str());
	EXPECT_STREQ("kkjluvip", messages[1].c_str());
	EXPECT_STREQ("of huck", messages[2].c_str());
	EXPECT_STREQ("zzzerrormsg", messages[3].c_str());
	EXPECT_STREQ("zzzxxxyyy", messages[4].c_str());
}


TEST(LOGGER, NoSupport)
{
	global::set_logger(new tutil::NoSupportLogger());

	auto logger = dynamic_cast<tutil::NoSupportLogger*>(&global::get_logger());
	global::infof("hello %s", "world");
	EXPECT_FALSE(logger->called_);
	global::debugf("hello %d", 0);
	EXPECT_FALSE(logger->called_);
	global::warnf("hello %d", 42);
	EXPECT_FALSE(logger->called_);
	global::errorf("hello %f", 0.0);
	EXPECT_FALSE(logger->called_);
	global::throw_errf("hello %f", 1.0);
	EXPECT_FALSE(logger->called_);
	global::fatalf("hello %f", 2.0);
	EXPECT_FALSE(logger->called_);

	global::set_logger(nullptr);
}


#endif // DISABLE_GLOBAL_LOG_TEST
