---
title: "The Big List of Slop"
summary: ""
date: 2026-11-12T12:00:00+00:00
tags: ["thoughts"]
type: post
showTableOfContents: true
image: "/2026-11-list-of-slop/spongebob-garbage.webp"
weight: 1
build:
 list: never
---


<style>
.post .post-content {
    margin-top: 0px;
}

.post hr {
    margin: 10px;
}

/* Got a lot of heading in this one. */
.post-content h1 {
    margin-top: 10px;
    margin-bottom: 0px;
}

.post-content h2 {
    margin-top: 10px;
    margin-bottom: 0px;
}
</style>

What is slop?
Slop, to use a somewhat loose definition, is code (or any writing) that is _statistically_ likely, but that does not provide value.

Of course, LLMs generate _more_ slop than humans by design<sup style="font-size: x-small;">[_citation needed_]</sup>. \
So while slop predates LLMs, there's much much more of it nowadays.

Since I still read the code, and since I expect to continue doing so (unless we all go extinct<sup style="font-size: x-small;">[_citation needed_]</sup>), I decided to collect common (and uncommon) slop-y patterns I have seen in the wild, and some remedies I found useful. 

So without further ado: Ohad's Big List of Slop. Starting with my current favorite, the MacGyver Class.

_Note: while the code in this post is obviously inspired by LLMs, no LLMs were used to generate anything - code or prose. 
Also, I'm thinking of updating this list as we invent (discover?) more interesting and strange forms of slop.
So send your slop my way at `slop@bikinibottom.io`! Gotta catch them all._

**Table of Contents**:

{{% toc %}}

# MacGyver Class

A modern take on NIH (Not Invented Here) meets a God class.

It takes both to the next level and implements weird low-level-ish primitives that are a combination of common building blocks that don't have a clear propose and are practically impossible to get right or hammer into a reusable shape.

```python
class LockableSocket:
   socket: socket.Socket
   lock: threading.Lock
```

No matter _what_ you are trying to do, the chances of someone **needing ∧ correctly building** something reusable with just a lock and a socket are virtually zero.

**Treatment**: Sometimes, you are the in the business of building lockable-socket-shaped-services, but if you are working with other people then it’s much more likely someone already built one. If not, maybe at least split into a different PR? \
More often, these are symptoms of a design bug. Fix the design! Don't document the lockable socket to convince me it's the next big thing in microservice management.

# Validate, Don’t Parse

The inverse of [Parse, don’t validate](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/),
it is code that does some preparation work that is required for correctness, but doesn't encode that (usually in some type-safe manner).

For example, say you make or read some list that needs to be sorted:

```python
def make_sorted_list():
    return [1,2,4]

data = make_sorted_list()
bisect(data, 3)
```

A classic slop move would be to _add a validation that the list is, indeed, sorted_:

```diff
 def make_sorted_list():
     return [1,2,4]

+def validate_list_is_sorted(input):
+    assert sorted(input) == input

 data = make_sorted_list()
+validate_list_is_sorted(data)
 bisect(data, 3)
```

Classic slop - no value, but correct and would even catch rogue versions of `make_sorted_list` which is probably why some TDD code looks like this.

**Treatment**: Depending on how expressive your language of choice is, either remove or replace with a typed version:

```rust
struct SortedVec<T>(Vec<T>);
fn make_sorted_list(...) -> SortedVec<u8> { ... }
fn bisect(data: SortedVec<u8>, needle: u8) -> usize { ... }
```

Other common forms: inserting an item to a dictionary and then checking the item is there, adding an item to a list and checking the list isn't empty,
doing string manipulation like trim and then validating the new string has a specific length, etc.

# Load-bearing Façade

In the before times, there's was an infamous design pattern, the [Facade](https://en.wikipedia.org/wiki/Facade_pattern). \
Every action that has >=1 steps can be Facade-d over:

```python
class Bar: ...
class Foo:
    bar: Bar

def make_bar() -> Bar: ...
def make_foo(bar: Bar) -> Foo: ...

# `get_foo` is a facade.
def get_foo() -> Foo:
    bar = make_bar()
    bar = bar.with_timeout(1)
    foo = make_foo(bar)
    foo.start()
    return foo
```

Facades always were almost-slop, but recent advances in slop generation pushed the envelop here.

What happens when a refactor causes `Foo` to no longer need a `Bar`?

```diff
 class Foo:
-    bar: Bar
+    baz: Baz
```

Well, it turns out that robots, like humans, have [cable box collection](https://blog.jim-nielsen.com/2026/hands-off-my-cables/) tendencies.

After all, `make_bar` is _load bearing_ (🙄): it is used in tests, it's still part of `get_foo`, `with_timeout` could have unexpected side effects. Etc, etc.
"Best keep it".

```python
def test_make_foo():
    bar = make_bar()
    assert make_foo(bar) is not None
```

**Treatment**: Exfoliate. Facades are common, making them a prime reason for keeping old ~cables~ code around for show.
Always check if the thing that claims to still need something is, itself, still needed.

# Paranoia Driven Tests

Tests! Tests are awesome. But how do you decide what to test?

We can always add another test that checks for things that _don't_ happen... but are these tests adding any value?

```python
def sum(a, b): ...

def test_sum():
    with pytest.raises(ValueError, match="must be a number"):
        sum("1", None)
    assert sum(1, 2) != 5
```

Or even more extreme - what about testing _the validation_ for things that _don't_ happen:

```python
def validate_stuff() -> str: ...

def test_validate_stuff():
    assert "boogieman" not in validate_stuff()
```

LLMs [popularized](https://tvtropes.org/pmwiki/pmwiki.php/Main/GenrePopularizer) this genre, but `Mock.assert_not_called` was shipped in Python 3.5 (2014) so it's hard to say who is the [Trope Maker](https://tvtropes.org/pmwiki/pmwiki.php/Main/TropeMakers) in this case.

**Treatment**: Each test must produce shareholder value. You are the shareholder. Delete tests that don't.

## Insanity Driven Tests

Turning the paranoia to 11, we can also test each and every line of a function:

```python
def do_request():
    ...
    request_took_metric.observe(took_ms) 
    logger.info("Request done.")

def test_request():
    do_request()
    assert request_took_metric.get_value() > 0
    assert logger.called_with(level=INFO, message="Request done.")
```

You would think that no ~sane~ human or bot would write this, but consider:

1. This was in the dataset.
2. I personally have seen this in old code.
3. There's a built in [`assertLogs`](https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertLogs) in `unittest`.
4. I have seen [much, much worse][frank-init].

[frank-init]: {{< ref "/posts/2025-04-19-frank.md" >}}

This is an endgame variety of the [Change-Detector Tests](https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html), they take TDD and pretend it’s the cold war and the soviets are sabotaging the codebase.

**Treatment**: Same as the pervious point - delete and move on.

# Sneaky Dict Side Dish

A subtle and sinister variation on the tried and tested Dict Soup:

```python
# Party like it’s 2005.
def foo(): return {"foo": 1}
def foo_with_db(): return {"foo": 1, "db": ...}
```

This went out of fashion years ago, but has made a comeback in the form of a _side channel_ for anything that’s too painful to add to the "real" API:

```python
# Party like it’s 2025.
def foo_with_extra_stuff(idx: int stuff: dict):
     return Foo(idx, db, stuff=stuff)

def process_bar(bar: Bar) -> (int, dict):
    ...
    return 1, {"task_took_ms": now() - t0, "missed": ...}
```

**Treatment**: Do or do not - if something is important it should be typed and documented and be given attention and effort,
but if it isn't it should be removed.

# Thesaurus Oriented Naming

Let’s say you look for

> a word for two things connected at the middle

A thesaurus would suggest Coupler, or a Junction, or Nexus, or Conduit.

They can link, couple, entwine, intersect, unite, and attach.

However, 

```python
def connect(source, dest) -> Connection: ...
```

Is _probably_ the only correct answer: there’s no need to open the thesaurus.

What if we need something that really really really have to have that specific connotation of "middle"?

```python
def connect_mid(source, dest) -> MiddleConnection: ...
```

But honestly? Unless there a non-middle connection as well, just call it `Connection`. Less is more.

# Pinching Keystrokes

Like the C developers of old, more recent vintages of LLMs were _also_ RL-ed to use less tokens.

```c
// Ahh, the good old days (?).
int *x, *p, *q, **dd;
```

There's also _some_ pretext for shortening long things like package names, which is how we got to this:

```python
import numpy as np # good, honorable, classy

import json as j # evil, lazy, rowdy
```

**Treatment**: Unless it is a common abbreviation that's well documented, always use the full name. Always.

# Parsing XML with Regexes

You can't [parse XML with a regex](xml-regex). 
But that didn't stop generations of `{beginners,LLMs}` from trying.

[xml-regex]: https://stackoverflow.com/questions/8577060/why-is-it-such-a-bad-idea-to-parse-xml-with-regex

Sometime it's just plucking a value from a JSON:

```python
data = '{"name": "Bob"}`
name = re.search(r'"name"\s*:\s*"([^"]*)"', data).group(1)
```

Sometimes it's a more sneaky version like an ad hoc string escaping with a `replace`:

```terraform
replace(
    var.some_url,
    var.some_str_template, 
    urlencode(
        replace(replace(replace(var.some_value, "\\", "\\\\"), "\"", "\\\""), "\n", "\\n")
    )
)
```

It might even cover all the cases in a specific scenario!

**Treatment**: Avoid, but like in the [MacGyver Class](#macgyver-class), these are usually _symptoms_ that something went wrong somewhere else, and fixing the root issue will probably help more than importing-the-right-library.

# Disposable Reusable {Closures,Structs,Interfaces}

TODO

```rust
let make_i32 = |v| { v as i32 };
return optional_string.map(|s| make_i32(&s))
```

```go
process := func(x int) {
    s.actuallyDoSomething(x)
}

for _, item := range items {
    process(item)
}
```

See also: disposable classes that have two fields are only used in one place, interfaces with one function that are only used by one module, 
