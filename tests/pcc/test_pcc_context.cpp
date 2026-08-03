// Compaction stops being destructive.
//
// This is the check the whole component exists for, and it is written as the scenario
// rather than as unit assertions: a run reads a file, the context fills, a trim
// summarizes six turns down to six anchor lines, and then -- twenty turns later -- the
// agent needs a detail that only ever existed in the text that trim deleted.
//
// Before this component that detail was unrecoverable. compact_oldest() printed "events
// 1-6" into the summary it left behind, which read like a provenance pointer and pointed
// at nothing.

#include <string>
#include <string_view>
#include <vector>

#include "src/context/context.hpp"
#include "src/pcc/recall.hpp"
#include "src/pcc/store.hpp"
#include "tests/check.hpp"

namespace {

using lmp::context::ContextStore;
using lmp::context::TurnRecord;
using namespace lmp::pcc;

lmp::pcc::TimeUs g_now = 1000;
lmp::pcc::TimeUs manual_clock() { return g_now; }

// The sidecar's wiring, in miniature (src/surface/context_journal.cpp).
//
// TURNS ARE WRITTEN AS THEY ARRIVE, not when a trim is about to destroy them. The
// compaction sink alone used to be the whole adapter, and it only fires once the prompt
// crosses the compaction threshold -- which on real work it often never does. Measured
// 2026-08-03: a workspace with several complete 80-turn runs through it had a context
// database with its schema and zero rows.
void journal_to(ContextStore& ctx, Store& store) {
    ctx.set_turn_sink([&store](const TurnRecord& turn) {
        Record rec;
        rec.kind = kind::kTurn;
        rec.first_event = turn.first_event_seq;
        rec.last_event = turn.last_event_seq;
        rec.title = turn.tool_name;
        rec.body = turn.user_text + turn.assistant_text +
                   (turn.observation.empty() ? "" : "\n" + turn.observation);
        store.append(std::move(rec));
        g_now += 100;
    });
    // Compaction now contributes the SUMMARY and nothing else: the turns behind it are
    // already rows, and appending them again would double every compacted turn.
    ctx.set_compaction_sink([&store](const std::vector<TurnRecord>& dropped,
                                     std::size_t span_index, std::string_view summary) {
        Record rec;
        rec.kind = kind::kSpan;
        rec.first_event = dropped.front().first_event_seq;
        rec.last_event = dropped.back().last_event_seq;
        rec.title = "compacted span " + std::to_string(span_index + 1);
        rec.body = std::string(summary);
        store.append(std::move(rec));
        g_now += 100;
    });
}

TurnRecord tool_turn(const std::string& tool, const std::string& args,
                     const std::string& observation, std::uint64_t event) {
    TurnRecord t;
    t.tool_name = tool;
    t.tool_args_summary = args;
    t.observation = observation;
    t.first_event_seq = event;
    t.last_event_seq = event;
    return t;
}

TEST(compaction_without_a_sink_is_unchanged) {
    // The sink is optional and nothing about the default path may move: a ContextStore
    // in a unit test still needs no database.
    ContextStore ctx("mission");
    for (std::uint64_t i = 1; i <= 6; ++i) {
        ctx.add_turn(tool_turn("read_file", "a.cpp", "line " + std::to_string(i), i));
    }
    CHECK_EQ(ctx.compact_oldest(2), std::size_t{4});
    CHECK_EQ(ctx.recent().size(), std::size_t{2});
    CHECK_EQ(ctx.compacted_spans().size(), std::size_t{1});
}

TEST(turns_are_durable_before_any_trim_happens) {
    // THE BUG THIS COMPONENT WAS BUILT WITH, and the reason the turn sink exists.
    //
    // Journalling used to hang entirely off compaction, so a run that never filled its
    // context window never wrote a row -- and the store whose whole job is to outlive the
    // window was empty for exactly the runs that had the most history to keep. No trim
    // happens anywhere in this test, which is the point.
    Store store(":memory:");
    store.set_clock(&manual_clock);
    ContextStore ctx("mission");
    journal_to(ctx, store);

    ctx.add_turn(tool_turn("read_file", "a.cpp",
                           "the retry budget is seven attempts, not three", 1));
    for (std::uint64_t i = 2; i <= 5; ++i) {
        ctx.add_turn(tool_turn("read_file", "a.cpp", "line " + std::to_string(i), i));
    }
    CHECK_EQ(ctx.compaction_count(), std::size_t{0});
    CHECK_EQ(ctx.recent().size(), std::size_t{5});
    CHECK_EQ(store.stats().items, std::int64_t{5});

    // And it is retrievable by content, not merely present. This is what a later session
    // actually does: it asks a question, having forgotten which turn held the answer.
    RecallRequest req;
    req.query = "retry budget attempts";
    const Recall out = recall(store, req);
    REQUIRE(!out.entries.empty());
    CHECK(out.text.find("seven attempts") != std::string::npos);
}

TEST(compaction_journals_the_full_text_before_dropping_it) {
    Store store(":memory:");
    store.set_clock(&manual_clock);
    ContextStore ctx("mission");
    journal_to(ctx, store);

    // The detail that matters survives only in the observation body -- the anchor line
    // the summary keeps is truncated to 200 characters and would lose it.
    const std::string detail =
        "temperature clamp lives at sampler.cpp:88 and rounds toward zero, which is why "
        "top_p=0 degenerates instead of erroring; the guard added in 6da59c5 covers the "
        "decompose loop but NOT the second offset, and the reproduction needs a "
        "four-byte sequence whose continuation byte is 0x80 exactly";
    ctx.add_turn(tool_turn("read_file", "src/model/sampler.cpp", detail, 1));
    for (std::uint64_t i = 2; i <= 6; ++i) {
        ctx.add_turn(tool_turn("shell", "cmake --build build",
                               "build ok " + std::to_string(i), i));
    }

    const std::size_t dropped = ctx.compact_oldest(1);
    CHECK_EQ(dropped, std::size_t{5});

    // The prompt kept a summary, and the summary is genuinely lossy -- that is the
    // premise, not a complaint.
    REQUIRE(ctx.compacted_spans().size() == 1);
    const std::string& span = ctx.compacted_spans().front();
    CHECK(span.find("events 1-5") != std::string::npos);
    CHECK(span.find("continuation byte is 0x80") == std::string::npos);

    // And the full text is one query away, reached by the range the span printed.
    const Recall out = rehydrate(store, 1, 5, 4000);
    CHECK_EQ(out.included, std::size_t{5});
    CHECK(out.text.find("continuation byte is 0x80") != std::string::npos);
    CHECK(out.text.find("rounds toward zero") != std::string::npos);
}

TEST(journalled_turns_are_searchable_after_the_trim) {
    // Rehydration needs the event range. Search does not -- which matters, because an
    // agent twenty turns later remembers what it was looking at, not which span number
    // the detail fell into.
    Store store(":memory:");
    store.set_clock(&manual_clock);
    ContextStore ctx("mission");
    journal_to(ctx, store);

    ctx.add_turn(tool_turn("search", "utf8proc", "utf8proc decompose has a null pointer "
                                                 "offset in the second branch",
                           1));
    for (std::uint64_t i = 2; i <= 8; ++i) {
        ctx.add_turn(tool_turn("list_dir", "src", "entries " + std::to_string(i), i));
    }
    ctx.compact_oldest(1);
    // Nothing of it is left in the prompt beyond one truncated anchor line.
    CHECK_EQ(ctx.recent().size(), std::size_t{1});

    RecallRequest req;
    req.query = "utf8proc null pointer offset";
    req.token_budget = 2000;
    const Recall out = recall(store, req);
    REQUIRE(!out.entries.empty());
    CHECK(out.text.find("null pointer offset in the second branch") != std::string::npos);
}

TEST(repeated_trims_accumulate_rather_than_overwrite) {
    // A long run trims many times. Each trim must add to the store; the failure worth
    // guarding is a sink keyed by span index that overwrites the previous span.
    Store store(":memory:");
    store.set_clock(&manual_clock);
    ContextStore ctx("mission");
    journal_to(ctx, store);

    std::uint64_t event = 1;
    for (int trim = 0; trim < 4; ++trim) {
        for (int i = 0; i < 5; ++i) {
            ctx.add_turn(tool_turn("shell", "step", "trim " + std::to_string(trim) +
                                                        " step " + std::to_string(i),
                                   event++));
        }
        ctx.compact_oldest(1);
    }

    CHECK_EQ(ctx.compaction_count(), std::size_t{4});
    // 20 turns added and every one of them is a row, INCLUDING the one still sitting in
    // the prompt -- that is the change: durability no longer waits for a trim, so the
    // invariant is "every turn is in the store" rather than the old "every turn is in
    // exactly one of the two places". Plus one span row per trim: 20 + 4 = 24.
    CHECK_EQ(store.stats().items, std::int64_t{24});
    CHECK_EQ(store.by_kind(kind::kTurn).size(), std::size_t{20});
    CHECK_EQ(store.by_kind(kind::kSpan).size(), std::size_t{4});
    CHECK_EQ(ctx.recent().size(), std::size_t{1});

    // The span rows carry the same event ranges as the turns they summarize, so a
    // rehydrate must not hand back the summary the model is already looking at.
    // events_between() filters to turns for exactly this reason.
    const Recall span_range = rehydrate(store, 1, 4, 4000);
    CHECK_EQ(span_range.included, std::size_t{4});
    CHECK(span_range.text.find("compacted span") == std::string::npos);

    // The earliest trim's content is still there, which is the property that fails first
    // if the sink is stateful in the wrong way.
    const Recall first = rehydrate(store, 1, 4, 4000);
    CHECK(first.text.find("trim 0 step 0") != std::string::npos);
    const Recall last = rehydrate(store, 16, 20, 4000);
    CHECK(last.text.find("trim 3") != std::string::npos);
}

TEST(check_framework_can_still_fail_here) {
    ContextStore ctx("mission");
    EXPECT_FAILING_CHECKS(2, {
        CHECK(ctx.compaction_count() == 5);
        CHECK_EQ(ctx.mission(), std::string("something else"));
    });
}

} // namespace
