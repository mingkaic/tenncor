#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "wire/omap.hpp"


#ifndef DISABLE_OMAP_TEST


using namespace testutil;


class OMAP : public fuzz_test {};


TEST_F(OMAP, PutGet_A000)
{
    std::string key = get_string(get_int(1, "key.size", {16, 64})[0], "key");
    size_t value = get_int(1, "value", {1, 2523})[0];
    size_t value2 = get_int(1, "value2", {2524, 4253})[0];

    wire::ordered_map<std::string, size_t> omap;

    wire::optional<size_t> nf = omap.get(key);
    EXPECT_FALSE((bool) nf);
    EXPECT_TRUE(omap.put(key, value));

    wire::optional<size_t> gotv = omap.get(key);
    ASSERT_TRUE((bool) gotv);
    EXPECT_EQ(value, *gotv);

    EXPECT_FALSE(omap.put(key, value));
    EXPECT_FALSE(omap.put(key, value2));
}


TEST_F(OMAP, PutHas_A001)
{
    size_t nkey = get_int(1, "key.size", {16, 64})[0];
    std::string key = get_string(nkey, "key");
    std::string key2 = get_string(nkey + 1, "key2");
    size_t value = get_int(1, "value", {1, 2523})[0];

    wire::ordered_map<std::string, size_t> omap;
    EXPECT_FALSE(omap.has(key));
    EXPECT_TRUE(omap.put(key, value));
    EXPECT_TRUE(omap.has(key));
    EXPECT_FALSE(omap.has(key2));
}


TEST_F(OMAP, PutReplace_A002)
{
    size_t nkey = get_int(1, "key.size", {16, 64})[0];
    std::string key = get_string(nkey, "key");
    std::string key2 = get_string(nkey + 1, "key2");
    size_t value = get_int(1, "value", {1, 2523})[0];
    size_t value2 = get_int(1, "value2", {2524, 4253})[0];

    wire::ordered_map<std::string, size_t> omap;

    EXPECT_FALSE(omap.replace(key, value));
    EXPECT_TRUE(omap.put(key, value));

    wire::optional<size_t> gotv = omap.get(key);
    ASSERT_TRUE((bool) gotv);
    EXPECT_EQ(value, *gotv);

    EXPECT_TRUE(omap.replace(key, value2));

    wire::optional<size_t> gotv2 = omap.get(key);
    ASSERT_TRUE((bool) gotv2);
    EXPECT_EQ(value2, *gotv2);
}


TEST_F(OMAP, Range_A003)
{
    size_t nkey = get_int(1, "key.size", {16, 64})[0];
    std::string key = get_string(nkey, "key");
    std::string key2 = get_string(nkey + 1, "key2");
    size_t value = get_int(1, "value", {1, 2523})[0];
    size_t value2 = get_int(1, "value2", {2524, 4253})[0];
    size_t value3 = get_int(1, "value2", {1, 4253})[0];

    wire::ordered_map<std::string, size_t> omap;

    ASSERT_TRUE(omap.put(key, value));
    ASSERT_TRUE(omap.put(key2, value2));

    std::vector<size_t> exorder = {value, value2};
    std::vector<size_t> goorder;
    std::copy(omap.begin(), omap.end(), std::back_inserter(goorder));
    EXPECT_ARREQ(exorder, goorder);

    ASSERT_TRUE(omap.replace(key, value3));

    std::vector<size_t> exorder2 = {value2, value3};
    std::vector<size_t> goorder2;
    std::copy(omap.begin(), omap.end(), std::back_inserter(goorder2));
    EXPECT_ARREQ(exorder2, goorder2);
}


#endif /* DISABLE_OMAP_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
