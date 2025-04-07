---
title: "BTrees, Inverted Indices, and a Model for Full Text Search"
date: 2025-04-10T00:00:00+00:00
tags: ["rust", "databases", "practices", "performance"]
summary: "Whenever I get into a new technology, I try to build myself a mental model of how it works. Let's do that for _full text search engines_, by writing a minimal implementation in Rust"
type: post
showTableOfContents: true
image: "/2025-04-08-btrees-and-mental-models/inverted_index.png"
weight: 1
---

Whenever I get into a new technology (A new database feature, a fancy distributed queue, an orchestration system), 
I try to build myself a _mental model_ of how it works.

Today, we'll build a mental model for _Full Text Search_ engines (along with some code in Rust), as done in database systems such as Elasticsearch, PostgreSQL and ClickHouse.

By the end, you'll understand some of the key features, limitations, and solutions common to all these systems - and even predict how they behave in different scenarios.

## A Primer on Text Analysis

Consider the following example. Say we have a few documents, each describing a programming language:

| Name       | Description                                                                                                             |
|------------|-------------------------------------------------------------------------------------------------------------------------|
| Python     | Python is a programming language that lets you work quickly and integrate systems more effectively.                     |
| TypeScript | TypeScript is a strongly typed programming language that builds on JavaScript, giving you better tooling at any scale.  |
| Rust       | Rust is a statically-typed programming language designed for performance and safety.                                    |

Full text search boils down to this: given a query like `performant typed language`, find:
(a) *which* documents match the query and 
(b) *rank* them by how well they match the query.

## Build vs. Read

To build a good model of a system it can be enough to _do the reading_: 
documentation and technical write-ups are great for getting all the information you need to build a model of a system in your head, 
but it’s important not to get lost in the details, especially since most of these will focus on a _very specific solution_,
while we care more about the _problem_ and the _solution space_.

Writing some (minimal, even really bad) code is a great way to gain a deeper insight into the core ideas behind a system, 
so let's write a tiny piece of code that can match documents to a query, and use that to predict how a real system might behave. \
We'll set up our code so that we have this data in memory, and ignore the many details required to actually store it [safely on disk](https://danluu.com/file-consistency/).

```rust
let docs = [
    "Python .. lets you work quickly ..",
    "TypeScript .. builds on JavaScript  ..",
    "Rust is .. designed for performance and safety ..",
    "C makes it easy to shoot yourself in the foot .."
];

type DocId = usize;
```

## A Simple Tokenizer

To find documents matching the query `performant typed language`, we can take each word in the query, and look for it in each of the documents. 
This will be horrifically slow, but it nudges us the right direction: if we also split the documents into words ahead of time, we might be able to do faster lookups.

In our example, we would consider each small part of the text to be a word, but for more general texts the actual “atoms” are harder to categorize.

<picture>
    <source srcset="/2025-04-08-btrees-and-mental-models/tokens_dark.png" media="(prefers-color-scheme: dark)">
    <img src="/2025-04-08-btrees-and-mental-models/tokens_light.png">
</picture>

In text analysis, these atoms are called _tokens_[^2]. For now, we'll just naively split the text every time we see a whitespace character:

```rust
fn tokenize(doc: &str) -> Vec<&str> {
    doc.split_ascii_whitespace().collect()
}

let doc = "Python is a programming language that lets you work quickly and integrate systems more effectively.";

// Prints `["Python", "is", "a", "programming", .. , "effectively."]`.
dbg!(tokenize(doc));
```

Phew! Some 10x engineering right there. 

[^2]: Not to be confused with _tokens in the LLM sense_, which are sequences of characters that are mapped to a specific number in an embedding space, and can be parts of words (See for example: [Tiktokenizer](https://tiktokenizer.vercel.app/), and `openai/tiktoken`'s [BPEs](https://github.com/openai/tiktoken?tab=readme-ov-file#what-is-bpe-anyway)).

## An Inverted Index

Instead of scanning every document for every token in the query (usually called a _term_), we will _tokenize the documents_ and store the tokens from each document in an _inverted index_:
the token will be the key, and the value will be the document ids that contain that token.

| Token       | Document IDs |
|-------------|--------------|
| better      | [1]          |
| concurrency | [2]          |
| typed       | [1, 2]       |
| language    | [0, 1, 2]    |
| performance | [2]          |
| ..          | ..           |

There's a small wrinkle here because `performant` and `performance` are not the same token, which is one of the things we'll handle in the next section.

## BTrees, the Powerhouse of the Database

We are going to use [`BTreeMap`](https://doc.rust-lang.org/std/collections/struct.BTreeMap.html) from the standard library as the data structure for our index.

For a _mental model_ of a full text search engine, we can treat all the minutiae around binary trees, b-trees, b+trees, etc. as, well, minutiae, 
and _think_ about the `BTreeMap` index as a **stored array with cheap insert/remove**, with values attached to the sorted elements - just like the table above (sorted arrays have cheap - in the `O(log(N))` sense - lookup using binary search, so we are getting this "for free" here).

This is _technically_ completely incorrect, but it's a very useful **approximation**. When working out a mental model, having a good approximation and knowing _when_ that approximation doesn't apply[^4] can be more useful than hauling out the precise formulation for every analysis.

[^4]: Most notably, this approximation fails when dealing with high insert or delete workloads: keeping the tree balanced is complex enough that most implementations will run compactions at certain intervals and only mark a node as deleted without actually removing it. 
Also, the actual relationship between the on-disk and in-memory representations (and how fast it is to load from / spill over to the disk) can have operational consequences.

The reason we want to think about a _map_ as a sorted array/table is because we'll be working with ranges. In our little example, `performant` and `performance` aren't the same token, but they do share a **stem**: if we stemmed the query and got back something like `perform`, we can use our inverted index to get all the tokens starting with `perform` efficiently by binary-searching for it, and then advancing along the values of the index.

In Rust, this is what building such an index looks like:

```rust
// Create a new index.
let mut index: BTreeMap<String, Vec<DocId>> = BTreeMap::new();

for (doc_id, doc) in docs.iter().enumerate() {
    let tokens = tokenize(doc);

    // Insert each token into the index, with the document id added to the value.
    for token in tokens {
        index
            .entry(token.to_string())
            .or_insert_with(|| vec![])
            .push(doc_id);
    }
}

// Prints `{"C": [3], "JavaScript,": [1], "Python": [0], "Rust": [2], "TypeScript": [1], .. }`.
dbg!(&index);
```

The call to `or_insert_with` will insert an empty `Vec` into the index for the `token` key if it's missing, and return the `Vec` (either the newly created one or the existing one) so we can `push` the `doc_id` into it.

Querying the index by key is simpler but we won't get any result for the `perform` term:

```rust
let original_query = tokenize("performant typed language");
// .. stemmed each term of the query, for example using a dictionary ..
let query = ["perform", "typed", "language"];

dbg!(index.get(query[0])); // None
dbg!(index.get(query[1])); // Some([1, 2])
dbg!(index.get(query[2])); // Some([0, 1, 2])
```

This brings us to the other way to query the `BTreeMap`, and the reason it's useful to approximate it as a sorted array: 
we can query it efficiently - again, in the `O(log(N))` sense - by _range_, which for `String`s will use lexicographic order:

```rust
let term = "perform".to_string();
for (key, value) in index.range(term..) {
    // key = "performance", value = [2]
    // key = "programming", value = [0, 1, 2]
    // ...
    dbg!(key, value);
}
```

Calculating the matching docs can be done by taking an intersection of all the docs ids returned for each term. See [Appx. 1](#appendix-1---compared-to-naive-search) for some additional perf analysis.

### Ranking

When searching for text in a large corpus, it's common to only want to return the "best" N results, since many more documents may match a given query than are actually useful to the user.

To meaningfully rank our results we need more data, but we can already build some intuition: We can see that if we count the total number of documents that contain each token, we see that `performance` (1) < `typed` (2) < `language` (3). We can deduce that `performance` is a more "specific" token, meaning that documents that contain it (especially multiple times) would be a better match for the query. 

This idea is called [tf–idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), or `term frequency-inverse document frequency`, and for our mental model it is enough that we think about it as "we also need to store some statistics about our corpus" in order to perform **basic** ranking. 

For example, Elasticsearch uses [BM25](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) by default, which is an advanced tf-idf-like ranking function. PostgreSQL has two ranking functions [`ts_rank` and `ts_rank_cd`](https://www.postgresql.org/docs/current/textsearch-controls.html#TEXTSEARCH-RANKING), and both are based on term frequency.

## Four Features and a Funeral 

Despite being frugal, we can use this model to predict some interesting properties a full text search system will have:

🔪 **Sharding** is almost free: \
Splitting documents across nodes in a cluster is straightforward with this architecture. Each node can perform text analysis and indexing [independently](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-replication.html#basic-write-model), so a system can scale horizontally. 

However, our model also reveals a tradeoff: when executing queries, each node will need to return the top N matches (not knowing what other nodes have found), and a single node (or the client) will need to rank the final results. Our model also shows that the longer the query (or, more precisely, the number of terms in the processed query), the more work the database will need to do to execute it - each term requires a lookup and an intersection operation. 

#️⃣ **Prefix** search is free, general wildcards are not: \
Our model shows that executing a prefix query like `python lang*` is essentially free: we can use the same `range` operation on the index to get the matching documents. \
Implementing **postfix** search can be done by reversing each token and index it a second time in a separate index, doubling the storage requirements. 

For example, Elasticsearch supports [`prefix` queries](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-prefix-query.html) which are just as fast as regular queries, and has a built-in [`reverse`](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-reverse-tokenfilter.html) option to index reversed tokens, and PostgreSQL supports prefix queries using a syntax like `lang:*` that can be passed to [`::tsquery`](https://www.postgresql.org/docs/current/datatype-textsearch.html#DATATYPE-TSQUERY).

But we can see from our model that supporting queries like `*ython* language` is a lot harder! To do it efficiently, we can use [_n-grams_](https://en.wikipedia.org/wiki/N-gram), which is like applying a sliding window tokenizer to the original text, and will increase the index size a lot. Elasticsearch has an [`ngram`](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-ngram-tokenizer.html) tokenizer, and ClickHouse supports `ngrams` via the [`full_text(ngrams)`](https://clickhouse.com/docs/engines/table-engines/mergetree-family/invertedindexes#usage) index creation option.

📊 The **Common tokens** problem: \
In our code, common tokens like `the` and `a` require storing an enormous number of document ids, yet they are almost useless for searching and ranking documents (counterpoint: consider a query like `bear` vs `The Bear`). [Filtering](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-stop-tokenfilter.html) such [stop words](https://en.wikipedia.org/wiki/Stop_word) specifically, together with stemming and tokenizing, requires [language specific](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html) knowledge.

Regular tokens also tend to have an uneven distribution - some (like `language` in our example) are very common, while many tokens are rare. We used a vector to store the document ids, which will be inefficient at both ends of this spectrum, so we can guess that different implementations will use more efficient data structures for them: For example, ClickHouse uses [roaring bitmaps](https://clickhouse.com/blog/clickhouse-search-with-inverted-indices#posting-lists) while PostgreSQL uses [`GIN`](https://www.postgresql.org/docs/current/textsearch-indexes.html#TEXTSEARCH-INDEXES) which is [implemented](https://www.postgresql.org/docs/current/gin.html#GIN-IMPLEMENTATION) as a B-tree of B-trees.

✂️ The **Multi-Field Filtering** Challenge: \
When users want to combine text search with structured filters (`"typescript btree" AND stars > 1000`), our model points to a problem: unless we create a dedicated index that uses both fields, we'll have to get all the doc ids matching the text and intersect them with the doc ids that match the other filter(s). PostgreSQL supports this via [multicolumn GIN indices](https://www.postgresql.org/docs/current/indexes-multicolumn.html), while Elasticsearch doesn't support creating such indices but instead focuses on having a very efficient intersection implementation.

🔴 Despite being super useful, there are two related features that are supported by most full text search engines, but are hard to support with our current model: highlights and followed-by queries. But not all is lost! By extending our model a little bit, we can solve these limitations elegantly.

## A Position is Worth a Thousand Cycles

Consider a query like `javascript language`: since the TypeScript document has tokens matching both terms, we will consider it a match for this query.
We can let the user specify that they want a more "exact" match, for example by using `"javascript language"`, meaning that the term `language` must _follow_ the term `javascript`.

Our index currently maps from tokens to document ids (`BTreeMap<String, Vec<DocId>>`), so we can find the superset of documents that match this query, and then **re-process** every one of them and figure out at what order the terms appear in them.

This can be absolutely fine! Our example documents are very short, and if we don't expect an exact query to match a lot of documents (or don't intend to support them at all), doing this extra work isn't that bad.

A similar analysis applies for generating highlights for our results:

> Query: `performant typed language` \
> Response: `rust`: "Rust is a statically-<mark>typed</mark> programming <mark>language</mark> designed for <mark>performance</mark> and safety."

Doing this "on the fly" for short documents can be fine, but if we have bigger documents (e.g. entire Wikipedia articles), we’ll end up paying a high compute cost for each query.

The key insight is that we need to know not just _which_ documents contain the terms, but _where_ those terms appear in each document.
We can do that by trading query-time compute with ingest-time compute and storage: 
instead of only storing the document **ids** for each token, we can also store its offsets in the document!

This will require us to re-structure both our index and tokenizer. 

We'll define a struct to hold the start and end offsets of a token:

```rust
type DocId = usize;

#[derive(Debug, Clone)]
struct TokenOffsets(usize, usize);
```

Returning a huge `Vec` is actually bad for performance, so we'll also let the tokenizer insert each token into the index.

We _could_ implement a small `struct` for the iteration and generation of tokens and offsets, but that require a significant amount of Rust-fu which is pretty irrelevant to our little experiment (you can peruse such an implementation at your leisure in [Appx. 2](#appendix-2---a-token-iterator-in-rustt)).

```rust
fn tokenize_and_insert(
    doc_id: usize,
    doc: &str,
    index: &mut BTreeMap<String, Vec<(usize, TokenOffsets)>>,
) {
    // .. snip ..
}
```

We'll track the current token's offsets using the new structure, and update it as we iterate over the document:

```rust
let split_chars = HashSet::from([' ', '-', '.', ',']);

let mut token_offsets = TokenOffsets(0, 0);

for (char_offset, char) in doc.chars().enumerate() {
    if split_chars.contains(&char) {
        token_offsets.1 = char_offset;
        
        // Get the slice of `doc` for the token offsets.
        let token = &doc[token_offsets.0..token_offsets.1];

        index
            .entry(token.to_ascii_lowercase())
            .or_insert_with(|| vec![])
            // This time, we store both the doc id and the offsets as values.
            .push((doc_id, token_offsets));
        
        // The next token will start at the next character.
        token_offsets = TokenOffsets(char_offset + 1, char_offset + 1);
    }
}
```

The result is that for a query like `"javascript language"` we get:

```rust
let query = ["javascript", "language"];

dbg!(index.get(query[0])); // Some([(1, TokenOffsets(67, 77))])
dbg!(index.get(query[1])); // Some([ .. ,  (1, TokenOffsets(43, 51)), .. ]
```

And since 67 > 43, we know that the document doesn't match the query, without accessing the original text!

Storing the token's offsets will allow us to do highlighting very quickly but will use a lot more storage. We could also use the _order_ of the tokens (i.e. 1, 2, 3, ..) 
which still allows us to perform follow-by queries efficiently, while using something like a `u16` to store the order (usually called `position`) in the index.

Another benefit of using positions is that we can support queries with "slop", meaning a query can allow some number of other tokens in the document between the query's terms.

For example, Elasticsearch provides a few different [`index_options`](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-options.html) for text fields: 
storing only the document ids, storing `positions` (the order of tokens in the document), or storing `offsets` (the actual offset of the token, for faster highlighting, as well as positions), and can [highlight](https://www.elastic.co/guide/en/elasticsearch/reference/current/highlighting.html) fields indexed in any of them (for a compute cost when executing the query). The [`match_phrase`](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-match-query-phrase.html) query supports a `slop` parameter.

PostgreSQL's `ts_vector` stores tokens and positions and queries support a special syntax (`<N>`) for followed-by with a specified slop, and has a [limitation](https://www.postgresql.org/docs/current/textsearch-limitations.html) of 16,383 positions per vector. 
PostgreSQL also provides the [`ts_headline`](https://www.postgresql.org/docs/current/textsearch-controls.html#TEXTSEARCH-HEADLINE) function, but notes that it "uses the original document, not a `tsvector` summary, so it can be slow and should be used with care".

## Summary

That's it folks! We have a model of a full text search engine - it boils down to splitting text into tokens, storing tokens in a sorted index with their document ids and position, and fetching by prefix. \
We even saw that the predictions we made based on this model matched the behaviors and documentation of different implementations.

I used this model _a lot_ back when I was working with big Elasticsearch clusters and needed to implement complex queries, custom mappings, and analyzers, 
and it helped me to better understand everything about the system, the things we observed when putting it under pressure, and avoid running expensive experiments.

Building an accurate model of a system can be sometimes be done purely by reading, but it can be easy to get lost in the details, and a small amount of code can help keep the focus on the most important parts.

Are there any particularly useful mental models you have? Discuss on [r/rust](https://www.reddit.com/r/rust/), [lobsters](https://lobste.rs/)! 👋

## Appendix 1 - Compared to Naive Search

We can compare our index to a naive search strategy: look for each term by scanning all the documents over and over.

To isolate the interesting parts, we'll skip returning the `Vec` with the document ids (which, as we saw, was the least realistic part of the implementation anyway).

We'll use [Criterion](https://github.com/bheisler/criterion.rs) to do the benchmarking.

Our naive search is simply:

```rust
fn search_using_str_find(query: &[&str], docs: &[&str]) {
    for &term in query {
        for (doc_id, doc) in docs.iter().enumerate() {
            if doc.find(term).is_some() {
                black_box(doc_id);
            }
        }
    }
}
```

While our index-based search will use:

```rust
fn search_using_index(query: &[&str], index: &BTreeMap<&str, Vec<DocId>>) {
    for &term in query {
        let (key, value) = index.range(term..).next().unwrap();

        if !key.starts_with(term) {
            continue;
        }

        black_box(value);
    }
}
```

Note that we're using a slightly different index type here: For our key, instead of using a `String` (heap-allocated vector of chars), we are using `&str` (a slice of chars). This is more realistic and allows us to remove an annoying allocation which would make the benchmark useless.

Using the same query as before, we can add a few more documents to see how performance is affected by more data.
```rust
let query = ["perform", "typed", "language"];
let docs = [
    "Python is a programming language that lets you work quickly and integrate systems more effectively.",
    "TypeScript is a strongly typed programming language that builds on JavaScript, giving you better tooling at any scale.",
    "Rust is a statically-typed programming language designed for performance and safety, especially safe concurrency and memory management.",
    "C makes it easy to shoot yourself in the foot. C++ makes it harder, but when you do, you blow your whole leg off.",
    "Java is a class-based, object-oriented language that is designed to be portable, secure, and run on billions of devices.",
    "Go is a simple, statically typed language designed by Google for efficient concurrency and fast compilation.",
    "JavaScript is the language of the web, enabling interactive frontends and full-stack development with Node.js.",
    "Kotlin is a modern language that runs on the JVM and is fully interoperable with Java, with cleaner syntax and null safety.",
    "Swift is Apple's language for iOS and macOS development, designed for safety, speed, and modern features.",
    "PHP is a server-side scripting language especially suited to web development, powering WordPress and much of the internet.",
];
```

And running our benchmarks:

```rust
c.bench_function("search using str find with 4 documents", |b| {
    b.iter(|| search_using_str_find(&query, &docs[..4]))
});

c.bench_function("search using str find with 10 documents", |b| {
    b.iter(|| search_using_str_find(&query, &docs))
});

c.bench_function("search using index with 4 documents", |b| {
    let index = build_index(&docs[..4]);
    b.iter(|| search_using_index(&query, &index))
});

c.bench_function("search using index with 10 documents", |b| {
    let index = build_index(&docs);
    b.iter(|| search_using_index(&query, &index))
});
```

Results in:

```bash
search using str find with 4 documents
                        time:   [398.36 ns 400.39 ns 403.18 ns]

search using str find with 10 documents
                        time:   [995.03 ns 1.0005 µs 1.0067 µs]

search using index with 4 documents
                        time:   [122.27 ns 122.74 ns 123.30 ns]

search using index with 10 documents
                        time:   [110.82 ns 112.13 ns 113.97 ns]
```

Which matches our expectations: naive search scales linearly with the number of documents, and is always slower than using an index (which is barely affected by the small increase in data).

## Appendix 2 - A Token Iterator in Rust

My kingdom for a `yield` keyword in Rust[^5]. But until then, setting up an iterator that returns offsets of tokens isn't too bad.

We can even get away with a single lifetime, and tracking a single additional variable for the current offset in the document:

```rust
#[derive(Debug, Clone)]
struct TokenOffsets(usize, usize);

struct TokenOffsetsIter<'a> {
    doc: &'a str,
    current_offset: usize,
    split_chars: &'a HashSet<char>,
}

impl<'a> Iterator for TokenOffsetsIter<'a> {
    type Item = TokenOffsets;

    fn next(&mut self) -> Option<Self::Item> {
        for (char_offset, char) in self.doc.chars().enumerate().skip(self.current_offset) {
            if self.split_chars.contains(&char) {
                let token_offsets = TokenOffsets(self.current_offset, char_offset);

                self.current_offset = char_offset + 1;

                return Some(token_offsets);
            }
        }

        None
    }
}
```

[^5]: Yes, I know this keyword is already reserved! You know what I mean 👀