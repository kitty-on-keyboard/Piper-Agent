#include <gtest/gtest.h>
#include "../include/markdown_stream.hpp"
#include <iostream>

using namespace md;
using namespace std;

void print_events(const vector<Event>& events) {
    for (const auto& ev : events) {
        switch (ev.kind) {
            case EventKind::Text: cout << "Text[" << ev.text << "] "; break;
            case EventKind::CodeBlockOpen: cout << "CodeBlockOpen[" << ev.info << "] "; break;
            case EventKind::CodeBlockText: cout << "CodeBlockText[" << ev.text << "] "; break;
            case EventKind::CodeBlockClose: cout << "CodeBlockClose "; break;
            case EventKind::InlineCodeOpen: cout << "InlineCodeOpen "; break;
            case EventKind::InlineCodeClose: cout << "InlineCodeClose "; break;
            case EventKind::HeadingOpen: cout << "HeadingOpen(" << ev.level << ") "; break;
            case EventKind::HeadingClose: cout << "HeadingClose "; break;
            case EventKind::ListItemOpen: cout << "ListItemOpen(" << ev.level << ") "; break;
            case EventKind::ListItemClose: cout << "ListItemClose "; break;
            case EventKind::ParagraphBreak: cout << "ParagraphBreak "; break;
        }
    }
    cout << endl;
}

vector<Event> merge_text_events(const vector<Event>& events) {
    vector<Event> merged;
    for (const auto& ev : events) {
        if (!merged.empty() && merged.back().kind == ev.kind &&
           (ev.kind == EventKind::Text || ev.kind == EventKind::CodeBlockText)) {
            merged.back().text += ev.text;
        } else {
            merged.push_back(ev);
        }
    }
    return merged;
}

vector<Event> get_all_events_one_chunk(string_view doc) {
    MarkdownStream ms;
    vector<Event> events = ms.feed(doc);
    auto end_events = ms.finish();
    events.insert(events.end(), end_events.begin(), end_events.end());
    return merge_text_events(events);
}

vector<Event> get_all_events_split(string_view doc, size_t split_index) {
    MarkdownStream ms;
    vector<Event> events = ms.feed(doc.substr(0, split_index));
    auto e2 = ms.feed(doc.substr(split_index));
    auto e3 = ms.finish();
    events.insert(events.end(), e2.begin(), e2.end());
    events.insert(events.end(), e3.begin(), e3.end());
    return merge_text_events(events);
}

void test_split_invariance(string_view doc, const string& name) {
    auto expected = get_all_events_one_chunk(doc);
    for (size_t i = 1; i < doc.size(); ++i) {
        auto actual = get_all_events_split(doc, i);
        EXPECT_EQ(expected, actual) << "Split invariance failed for document '" << name << "' at split index " << i << ". Char: '" << doc[i] << "'";
    }
}

TEST(MarkdownStreamTest, SplitInvariance_Basic) {
    test_split_invariance("Hello world", "Basic");
}

TEST(MarkdownStreamTest, SplitInvariance_Headings) {
    test_split_invariance("# Heading 1\n## Heading 2\n### Heading 3", "Headings");
}

TEST(MarkdownStreamTest, SplitInvariance_Lists) {
    test_split_invariance("* Item 1\n* Item 2\n  * Subitem 1", "Lists");
}

TEST(MarkdownStreamTest, SplitInvariance_CodeBlocks) {
    test_split_invariance("```cpp\nint x = 1;\n```\n", "CodeBlocks");
}

TEST(MarkdownStreamTest, SplitInvariance_InlineCode) {
    test_split_invariance("Some `inline code` here.", "InlineCode");
}

TEST(MarkdownStreamTest, SplitInvariance_AdversarialFences) {
    test_split_invariance("```cpp\nInside code block `not inline` \n```\nOutside", "AdversarialFences1");
    test_split_invariance("```\n```\n", "AdversarialFences2");
    test_split_invariance("``\nnot a code block\n``", "AdversarialFences3");
}

TEST(MarkdownStreamTest, UnterminatedCodeBlock) {
    MarkdownStream ms;
    auto e1 = ms.feed("```cpp\nint x;");
    auto e2 = ms.finish();
    vector<Event> events = e1;
    events.insert(events.end(), e2.begin(), e2.end());
    events = merge_text_events(events);

    // Check that it's closed and the code block text is intact
    ASSERT_GE(events.size(), 3);
    EXPECT_EQ(events[0].kind, EventKind::CodeBlockOpen);
    EXPECT_EQ(events[0].info, "cpp");
    EXPECT_EQ(events[1].kind, EventKind::CodeBlockText);
    EXPECT_EQ(events[1].text, "int x;");
    EXPECT_EQ(events[2].kind, EventKind::CodeBlockClose);
}

TEST(MarkdownStreamTest, HoldbackBound) {
    // Holdback limit test
    MarkdownStream ms;
    string large_input(2048, '`'); // Exceeds the ~1KB holdback bound
    auto events = ms.feed(large_input);
    EXPECT_FALSE(events.empty()) << "Stream should have flushed some events when holdback limit is exceeded.";
    ms.finish();
}

TEST(MarkdownStreamTest, ComprehensiveDocument1) {
    string doc = R"(# Markdown features
Here is a list:
* Item A
* Item B
  * Sub B1

```python
def foo():
    pass
```
Some `inline` code.
)";
    test_split_invariance(doc, "Comprehensive1");
}

// Additional docs for the ~20 representative docs requirement
string docs[20] = {
    "# Doc 0\nText\n",
    "## Doc 1\nMore text\n",
    "### Doc 2\n* List\n",
    "#### Doc 3\n1. Ordered\n",
    "##### Doc 4\n`Inline`\n",
    "###### Doc 5\n```\nBlock\n```\n",
    "Doc 6\n* A\n  * B\n",
    "Doc 7\n1. A\n  1. B\n",
    "Doc 8\n```cpp\nC++\n```\n",
    "Doc 9\n```python\nPython\n```\n",
    "Doc 10\n```rust\nRust\n```\n",
    "Doc 11\n```go\nGo\n```\n",
    "Doc 12\n```js\nJavaScript\n```\n",
    "Doc 13\n```ts\nTypeScript\n```\n",
    "Doc 14\n```java\nJava\n```\n",
    "Doc 15\n```c\nC\n```\n",
    "Doc 16\n```c#\nC#\n```\n",
    "Doc 17\n```ruby\nRuby\n```\n",
    "Doc 18\n```php\nPHP\n```\n",
    "Doc 19\n```swift\nSwift\n```\n",
};

TEST(MarkdownStreamTest, TwentyRepresentativeDocs) {
    for (int i = 0; i < 20; ++i) {
        test_split_invariance(docs[i], "Doc " + to_string(i));
    }
}
