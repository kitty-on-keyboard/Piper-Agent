// The model lifecycle: load, refuse, unload (M2).
//
// This is here because of a specific failure, and it is worth stating plainly. Loading
// the weights used to be a side effect of the first mission -- no method of its own, no
// state, no way for a surface to know whether the sidecar held 19 GB or nothing. So "no
// model is loaded" and "the model is thinking" rendered identically: a status line, and
// no output. A user sent a prompt into a freshly opened editor, was told it was thinking,
// and waited on a process that had never opened a checkpoint.
//
// Everything below is model-free on purpose. Both refusals happen at the tokenizer, which
// is a file read, so these assertions hold identically on a machine with MLX and one
// without -- and the gate has no model (S11.1).

#include <memory>
#include <string>

#include "src/platform/clock.hpp"
#include "src/surface/session.hpp"

#include "tests/check.hpp"

using lmp::surface::load_model;
using lmp::surface::ModelLoad;
using lmp::surface::Session;
using lmp::surface::unload_model;

namespace {

lmp::platform::ManualClock clock_;

} // namespace

TEST(an_empty_model_dir_is_refused_rather_than_defaulted) {
    // S7.5: no security-relevant input gets a default. An empty model_dir that fell
    // through to "some checkpoint" would be the whole rule going the wrong way.
    Session session;
    const ModelLoad r = load_model(session, "", "", clock_);
    CHECK(!r.ok);
    CHECK(!r.error.empty());
    CHECK(!session.model_ready());
}

TEST(a_missing_checkpoint_refuses_with_the_loaders_own_words) {
    Session session;
    const ModelLoad r = load_model(session, "/nonexistent/checkpoint", "", clock_);
    CHECK(!r.ok);
    // The path is IN the message. A load failure is nearly always a fixable statement
    // about what is on disk, and "load failed" with the path paraphrased away is the one
    // version of this the operator cannot act on.
    CHECK(r.error.find("/nonexistent/checkpoint") != std::string::npos);
    CHECK(!session.model_ready());
}

TEST(a_failed_load_leaves_no_half_loaded_session) {
    // The dangerous shape is a tokenizer for one checkpoint and weights for another:
    // that generates fluent nonsense instead of failing. model_ready() is both halves,
    // and holds() is what the sidecar asks before deciding whether a load will block.
    Session session;
    (void)load_model(session, "/nonexistent/checkpoint", "", clock_);
    CHECK(!session.model_ready());
    CHECK(session.model_dir.empty());
    CHECK(!session.holds("/nonexistent/checkpoint"));
    CHECK(!session.holds(""));
}

TEST(unload_takes_the_conversation_with_the_weights) {
    // A ContextStore whose model has gone is not resumable, so unloading is a full reset
    // of everything downstream of the weights rather than a free() of the weights alone.
    // If ctx survived here, the next prompt would be sent as a follow-up over a session
    // the sidecar can no longer answer for.
    Session session;
    session.model_dir = "/some/checkpoint";
    session.mcp_signature = "sig";
    session.ctx = std::make_unique<lmp::context::ContextStore>("a mission");
    unload_model(session);
    CHECK(!session.model_ready());
    CHECK(session.model_dir.empty());
    CHECK(session.ctx == nullptr);
    CHECK(session.registry == nullptr);
    CHECK(session.mcp_signature.empty());
}

TEST(unload_is_safe_on_a_session_that_never_loaded) {
    // The state a fresh process is in, and the state the surface can ask to unload from
    // -- a button that is only correct when something is loaded is a button that crashes.
    Session session;
    unload_model(session);
    CHECK(!session.model_ready());
}
