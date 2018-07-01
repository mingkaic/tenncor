#ifndef DISABLE_MOLD_MODULE_TESTS

#include "gtest/gtest.h"

#include "mold/ondeath.hpp"
#include "mold/variable.hpp"


#ifndef DISABLE_ONDEATH_TEST


TEST(ONDEATH, Trigger_F000)
{
    bool victim_killed = false;
    bool suicider_killed = false;
    mold::Variable* killer = new mold::Variable();
    mold::OnDeath* suicider = new mold::OnDeath(killer,
    [&]()
    {
        suicider_killed = true;
        suicider = nullptr;
    });
    delete suicider;
    EXPECT_TRUE(suicider_killed) << "suicide on death not executed";

    mold::OnDeath* victim = new mold::OnDeath(killer,
    [&]()
    {
        victim_killed = true;
        victim = nullptr;
    });
    delete killer;
    EXPECT_TRUE(victim_killed) << "victim on death not executed";
}


TEST(ONDEATH, Constructor_F001)
{
    bool clone_killed = false;
    bool mover_killed = false;
    mold::Variable* arg = new mold::Variable();
    mold::OnDeath* suicider = new mold::OnDeath(arg,
    [&]()
    {
        suicider = nullptr;
    });
    mold::OnDeath* clone = new mold::OnDeath(*suicider,
    [&]()
    {
        clone = nullptr;
        clone_killed = true;
    });
    mold::OnDeath* mover = new mold::OnDeath(std::move(*suicider),
    [&]()
    {
        mover = nullptr;
        mover_killed = true;
    });

    delete clone;
    EXPECT_TRUE(clone_killed) << "clone on death not executed";

    delete mover;
    EXPECT_TRUE(mover_killed) << "mover on death not executed";

    delete suicider;
    delete arg;
}


TEST(ONDEATH, ClearTerm_F002)
{
    bool suicider_killed = false;
    mold::Variable* arg = new mold::Variable();
    mold::OnDeath* suicider = new mold::OnDeath(arg,
    [&]()
    {
        suicider = nullptr;
        suicider_killed = true;
    });
    suicider->clear_term();

    delete suicider;
    EXPECT_FALSE(suicider_killed) << "cleared suicider on death was executed";

    delete arg;
}


#endif /* DISABLE_ONDEATH_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
