---
title: "Let Them Write RFCs"
date: 2026-08-08T04:30:00+00:00
tags: ["vibecoding", "thoughts"]
type: post
showTableOfContents: false
# image: "/2026-08-let-them-write-rfcs/XXX.webp"
weight: 1
build: 
 list: never
math: false
---

<style>
.post .post-content {
    margin-top: 0px;
}

.post hr {
    margin: 10px;
}
</style>

Or: why I spent hundreds of millions of tokens to write four versions of the same markdown file.

-----

I'm working on a complex project that's _just_ at the edge of what I know how to build, which is pretty fun! But also hard, especially when I try to speed things up by using LLMs.[^1] As a result:

1. I sometimes don't understand the problem well enough.
2. I sometimes don't _know_ that.
3. I'm more easily fooled by ideas that [sound good][pg-good-writing] due to 1 & 2.

[pg-good-writing]: https://www.paulgraham.com/goodwriting.html

[^1]: This project is an example for a time when [talking to the tool]({{< ref "/posts/2026-06-tool-talking.md" >}}) is _draining_ but totally worth it.

What I came up with is asking, sometimes after intensive [rambling][ramble-tweet], "... this is the vague problem we have right now. **Write a markdown file called $PROBLEM_RFC.md** about this problem and how to solve it. _Keep the prose short, have good code examples_."

Then, I'll **iterate and revise** until I'm convinced that I _do_ understand the problem well enough and the ideas _are_ sound. Big enough changes result in new $PROBLEM_RFC_V{N+1}.md file(s).

[ramble-tweet]: https://x.com/guinnesschen/status/2068744472528314811

I found it both **more useful**, as in producing better solutions and code, as well as **more enjoyable** for me to work on: instead of reading slop code, iterating on an RFC forces me to explain what I don't understand and to focus on hardest parts of the problem.

That's almost all there it to it - you should give it a try! If you want a template you can point the LLM at something like [Rust's Match Ergonomics RFC][match-ergo-rfc].

[match-ergo-rfc]: https://github.com/rust-lang/rfcs/blob/master/text/2005-match-ergonomics.md

You can read on for some reasons why I _think_ this worked unexpectedly well for me, but seriously you'll probably learn more by trying it.

### Why RFCs

The most important goal of an RFC is to **convince me** of a few different things:

1. What _exactly_ is the problem.
2. How bad / important it is.
3. Will this solution fix the problem.
4. How costly will the solution be, vs. the alternatives.

If I'm not convinced by any of these, especially by the definition of the problem, I'll just ask. "Explain this part of the problem", "Why is it this and not that?", "What about ...", "Didn’t we already do this in ..." - again and again until I’m actually convinced. 

I'll also usually have at least one round of asking - in a fresh context window - for an adversarial review, which needs to convince me of the reverse.

**An RFC is unlikely to be good enough on the first try**, and I usually end up with at least three or four versions.
The final version will change at least one but usually two major things from the initial version.

Also, RFCs:

1. Are faster and lower cost to produce than having the model write the full implementation.
2. Are faster and easier for me to review and spot problems.
3. Make it much more easy for me to spot problems-with-the-problem, which is a thing that's extremely hard to spot in an implementation.
4. Are cheap to have multiple versions of at the same time (V1.md, V2.md, etc), so I can go back and compare or change my mind and mix previous versions if the current version went bad, without git wizardry and [running out of disk space](https://bun.com/blog/bun-in-rust#false-starts).
5. Are usually mostly self-contained, so can be reviewed by other LLMs (Gemini 3.1 Pro, sometimes Opus/Fable) at a fraction of the cost.
6. Have a high standard of both prose and explanatory code, forcing more work on the LLM vs on the reader.
7. Have no conflicts when working on multiple things simultaneously: multiple RFCs can be written at the same time, _based on a current version that's not even compiling_, without interruptions.

Models have an inherit understanding of what an RFC is and how it should look.
I sometimes ask for the _style_ of the Rust project format (again, see [Rust's Match Ergonomics RFC][match-ergo-rfc]) but the actual structure should be flexible enough to fit what the problem requires - sometimes more motivation, more detailed design, performance, effects on different subsystems, an appendix for interactions with other features, etc. 

A good RFC is around 1k lines for the main content. The linked example is just over 500 lines including an alternatives section, but doesn't talk about the implementation. 1k lines is _a lot_ of room.

I usually commit the RFCs in the branch, but **I don’t merge them**. 
There’s a constant risk of LLMs-talking-to-LLMs that can be amplify random ideas/directions/limitation/words, and having a bunch of fat juicy markdown files lying a around is like catnip for the bots.

### Throw Away Code Anyway

Sometimes an RFC will sound great on paper (thanks Fable!) but the resulting code is garbage - a million edge cases, huge unexpected refactor, multiple modules dealing with a flow no one care about, etc.

With an RFC, I can say "this and that went wrong, read the diff, read the RFC, write a new version that ...", or I can even ask "this code is garbage, heres the RFC, what went wrong? Write an markdown with a review", or "this code solves 100 different edge cases, could we solve only 4 and have a clear error for the rest?" (and it sometimes turns out that 10 edge cases really were needed, but we can avoid the 90).

### vs. Plans

Usually, an RFC is not an actual implementation plan (like a plan in Cursor or plan mode in Claude code):
A plan is usually about *how* to hammer out the final code.

For a plan, I mostly care that all the things I asked for are accounted for (and their tests. Never forget the tests!),
and I probably know where and how the new code will fit with the existing code,
so it’s more of a way to spend more tokens on high-level work before kicking off the grind work.

### Read the Code

Of course I read the f*cking code.

```rust {linenos=inline,linenostart=2306}
fn column_context(side: &str) -> &str {
    if side == "input" { "input" } else { side }
}
```

I don’t read every single line, but I go over each file and most functions/structs. Yes it's hard in a 5k line PR. 
No there is - currently - no alternative.

