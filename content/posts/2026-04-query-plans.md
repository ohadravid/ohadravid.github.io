
---
title: "The Best (Query) Plans of Mice and Men"
date: 2026-04-26T00:00:00+00:00
tags: ["databases", "performance", "python"]
type: post
showTableOfContents: true
image: ""
weight: 1
build:
 list: never
---
<style>
.post .post-content {
    margin-top: 0px;
}
</style>
Or rather, of elephants 🐘 and men.

At [$work](https://www.wiz.io/blog/google-closes-deal-to-acquire-wiz), we use PostgreSQL _a lot_. \
So much in fact, that we tend to be ~paranoid~ frugal when _adding even more stuff_ to it.

In this case, we wanted to **add a new index** to speed up a query with an eye watering p99 latency of several seconds.
We have a single, big table that is shared between a few different features, but we are only interested in a tiny subset of rows from that table:

```go
// Less than %0.1 of rows match this.
db.Where("item_type = ?", 4)
```

~Being paranoid~ Having learned from past mistakes, we opted to create a _partial index_ instead of a full index:

```sql
CREATE INDEX CONCURRENTLY ix_special_item ON .. 
WHERE item_type = 4
```

Cheap to build, cheap to update, and cheap to store. \
But it got me wondering... can we know for certain that the DB will in fact use this new index?

What do you think? And would the answer change if instead the query was written like so:

```go
db.Where("item_type = 4")
```

TL;DR - Surprisingly, yes! That is because PostgreSQL, under [_certain conditions_][pg-docs-plan-cache-mode], will calculate and reuse query plans *without considering the specific parameters* that are being used, which could defeat the point of our clever index.

[pg-docs-plan-cache-mode]: https://www.postgresql.org/docs/current/runtime-config-query.html#GUC-PLAN-CACHE-MODE

To understand why, we'll need to first trigger this behavior - which requires setting up a schema with some data, some queries and indices (or you can jump straight [to the summary](#summary)). 

Now is a good time for a fresh cup of tea! 🍵

## Setting Up Our Tables and Data

Let's switch from Go with GORM to Python with SQLAlchemy, using the [Core API][insert-docs]. We do not need a full ORM for this example, and anyway ORMs are not _in vogue_.

Our example will model a backend for a restaurant order management app, using two tables. A small employee table:

```python
metadata_obj = MetaData()
employee_table = Table(
    "employee",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("name", String(32), nullable=False, unique=True),
    Column("email_address", String(60), nullable=True),
)
```

Into which we can [`insert()`][insert-docs] a few chosen employees:

```python
stmt = insert(employee_table).values(
    name="spongebob",
    email_address="spongebob@bikinibottom.io",
)
result = conn.execute(stmt)
conn.commit()
spongebob_id, = result.inserted_primary_key
```

[insert-docs]: https://docs.sqlalchemy.org/en/21/tutorial/data_insert.html#the-insert-sql-expression-construct

_Side note: yes I bought `bikinibottom.io` for this blog post. It could not have been helped._

And a soon to be huge `orders`[^2] table:

[^2]: Yes, using plural form is worse _and_ inconsistent with the `employee` table, but we don't want to use `order` to avoid confusion with `ORDER BY`.

<style>
/* Use a responsive grid to show the clippings side by side if the screen is wide enough */
.wrapper {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-gap: 2px;
}

.wrapper pre {
  margin: 0;
}

.box {
  padding: 2px;
}

.asm-intro-top-bottom {
  display: none;
}

@media (max-width: 520px) {
  .wrapper {
    grid-template-columns: 1fr; /* two rows */
  }
}
</style>

<div class="wrapper">
    <div class="box">
{{< markdownify >}}
```python
from enum import Enum
class KrabbyPattyItemType(Enum):
    KrabbyPatty = 1
    CoralBites = 2
    KelpRings = 3
    Special = 4

class Status(Enum):
    Pending = 1
    InProgress = 2
    Done = 3
```
{{< /markdownify >}}
    </div>
    <div class="box">
{{< markdownify >}}
```python
from sqlalchemy import Enum
orders = Table(
    "orders",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("item_type", Enum(KrabbyPattyItemType), ...),
    Column("status", Enum(Status), ...),
    Column("made_by", Integer, ForeignKey("employee.id"), ...),
    Column("timestamp", DateTime, ...),
    Column("item_details", JSONB, ...),
)
```
{{< /markdownify >}}
    </div>
</div>


Now we can fill it with 22,000[^3] Krabby Patties and Coral Bites (times 100), spread over a few years - some orders will be `InProgress` and some will be `Done`:

```python
# Spread times from first show air date to... 2069.
def choose_dt(i: int, j: int):
    return dt.datetime(1999, 5, 1) + dt.timedelta(seconds=i * 100000 + j * 10)

for i in range(22_000):
    rows = [
        {
            "made_by": spongebob_id,
            "status": Status.Done if i % 2 == 0 else Status.InProgress,
            "item_type": KrabbyPattyItemType.KrabbyPatty,
            "timestamp": choose_dt(i, j),
        }
        for j in range(100)
    ]

    stmt = insert(orders)
    result = conn.execute(stmt, rows)
    conn.commit()

# same for `CoralBites`.
```

This is not the fastest way to fill up a table<sup style="font-size: x-small;">[_citation needed_]</sup>,
but on my MacBook it takes ~2 minutes so it's good enough. 

[^3]: 
    Why 22,000? Well, the only AI usage (aside from spellchecking) in this article was using ChatGPT's Deep Research to "Estimating Krabby Patty Production Across SpongeBob SquarePants":

    > The most defensible series-wide estimate is about 25,000 Krabby Patties made on-screen or strongly implied on-screen, with a hard lower bound of about 8,100 and a liberal upper bound of about 43,000. 
    > 
    > On attribution, SpongeBob is still overwhelmingly the main producer. **My best attribution is about 22,000 patties made by SpongeBob himself**, ..

    So 22,000 it is. Good Bot! 🤖


We'll finish up by inserting a few specials from the [List of Krabby Patty variations] in _Encyclopedia SpongeBobia_:

[List of Krabby Patty variations]: https://spongebob.fandom.com/wiki/List_of_Krabby_Patty_variations

```python
insert(orders).values(
    made_by=spongebob_id,
    status=Status.InProgress,
    item_type=KrabbyPattyItemType.Special,
    item_details={
        "name": "Krusty Krab Pizza",
    },
)

insert(orders).values(
    made_by=spongebob_id,
    status=Status.InProgress,
    item_type=KrabbyPattyItemType.Special,
    item_details={
        "name": "Triple Krabby Supreme",
    },
)
```

Andddd done! We now have all the data we need - but we also need to query this data.

```sql
postgres=# SELECT * FROM orders LIMIT 2;
   id   |  item_type  | status | made_by |      timestamp      | item_details
--------+-------------+--------+---------+---------------------+--------------
 251201 | KrabbyPatty | Done   |       1 | 2007-04-16 09:46:40 |
 251202 | KrabbyPatty | Done   |       1 | 2007-04-16 09:46:50 |
(2 rows)
postgres=# SELECT count(*) FROM orders;
  count
---------
 4400002
(1 row)
```

<p style="text-align:center;">
    <img src="/2026-04-query-plans/Just_One_Bite_165.webp" 
        alt="Squidward looking into the the krabby patty vault in the Just One Bite episode" loading="lazy" 
        style="width:66%; height:auto;" width="1422" height="1080" />
</p>


## Our Query Function

We have 3 queries we want to perform against these tables - you can think of them as separate pages we want to display in our "Krusty Krab App":

1. Query the latest `InProgress` orders, regardless of `item_type`.
2. Query the latest `InProgress` orders for `KrabbyPatty` items.
3. Query the latest `InProgress` orders for `Special` items.

This 3rd use case represents our original problem: as we'll see, 
this query will be slower, and adding a full index is wasteful because we only care about a fraction of the rows in the table.

Using SQLAlchemy's [`select()`][select-docs], we can implement all 3 in a single simple Python function:

[select-docs]: https://docs.sqlalchemy.org/en/21/tutorial/data_select.html#the-select-sql-expression-construct

```python
def query_latest_in_prog(conn: Connection, limit: Optional[int] = 100, item_type: Optional[KrabbyPattyItemType] = None):
    # Base select: orders that are `InProgress`, with employee email.
    stmt = (
        select(orders, employee_table.c.email_address)
            .join(employee_table)
            .where(orders.c.status == Status.InProgress)
    )
    
    # Add the item type filter if provided.
    if item_type is not None:    
        stmt = stmt.where(orders.c.item_type == item_type)

    # Add a limit and order from latest.
    if limit is not None:
        stmt = stmt.limit(limit)
    stmt = stmt.order_by(orders.c.timestamp.desc())
   
    return list(conn.execute(stmt))
```

Now, 4.5 million rows isn't _that_ much, and computers are really fast. So, let's spin up a `jupyter lab` notebook and quickly measure[^1]:

[^1]: The `%%timeit -n` syntax tells the notebook to run the cell multiple times in a loop and measure the average time it took across multiple runs. In this case, 7 runs with 5 loops each.

```python
%%timeit -n 5
query_latest_in_prog(conn)
# 124 ms ± 2.27 ms per loop
```

Big ooof. Let's create a simple, regular index to speed this up:

```sql
CREATE INDEX ix_timestamp_only ON orders (status, timestamp)
```

The table takes about 200MB of storage and this new index requires an additional 100MB, but as you might expect:

```python
%%timeit -n 5
query_latest_in_prog(conn)
# 1.51 ms ± 208 μs per loop
```

Much better! and filtering for Krabby patties is almost as fast:

```python
%%timeit -n 5
query_latest_in_prog(conn, item_type=KrabbyPattyItemType.KrabbyPatty)
# 1.6 ms ± 141 μs per loop 
```

But... our poor Specials page!

```python
%%timeit -n 5
query_latest_in_prog(conn, item_type=KrabbyPattyItemType.Special)
# 151 ms ± 90.7 ms per loop
```

## A Partial Index

Simplified, we want the following query to execute faster:

```python
(
    select(orders)
        .where(orders.c.status == Status.InProgress)
        .where(orders.c.item_type == KrabbyPattyItemType.Special)
)
```

Or: XXX decide what to keep / keep both in "tabs"?
```sql
SELECT * FROM orders 
WHERE orders.status = 'InProgress' AND orders.item_type = 'Special' 
ORDER BY orders.timestamp DESC
```

But without paying for a full index, because `Special` items are only a tiny fraction of the rows in the `order` table.

So let's create a [partial index][pg-docs-partial-idx]: we'll still use the `timestamp` column, 
but add a filter so that only items that are `Special` are included:

[pg-docs-partial-idx]: https://www.postgresql.org/docs/current/indexes-partial.html

```sql
CREATE INDEX ix_timestamp_item_type_special ON orders (timestamp)
WHERE item_type = 'Special'
```

This new index is a few KB in size - much better than the 132MB a full `(item_type, timestamp)` index would consume - and as expected, doesn't affect the other queries:

```python
%%timeit -n 10
query_latest_in_prog(conn, item_type=KrabbyPattyItemType.KrabbyPatty)
# 2.14 ms ± 401 μs per loop
```

But is our query fast now?

```python
%%timeit -n 5
query_latest_in_prog(conn, item_type=KrabbyPattyItemType.Special)
# 238 ms ± 2.41 ms per loop
```

Not at all! Wat? Maybe we didn't setup the index correctly? Let's double check using a raw query:

```python
def query_ref(conn: Connection):
    result = conn.execute(text("""
SELECT *  
FROM orders JOIN employee ON employee.id = orders.made_by 
WHERE orders.status = 'InProgress' AND orders.item_type = 'Special' ORDER BY orders.timestamp DESC
LIMIT 100
"""))
    return list(result)
```

And...

```python
%%timeit -n 5
query_ref(conn)
# 417 μs ± 134 μs per loop
```

What? It _is_ faster? But... isn't it the same query?

XXX corporate needs you to tell the difference between XXX

## But Why?

So, what is happening here? To find out, let's take an SQL statement 

```python
stmt = (
    select(orders)
        .where(orders.c.status == Status.InProgress)
        .where(orders.c.item_type == KrabbyPattyItemType.Special)
)
```

and compile it:

```python
from sqlalchemy.dialects import postgresql
compiled_stmt = stmt.compile(dialect=postgresql.dialect())
```

What we get back is perhaps not exactly what you would expect. We get back a _parameterized_ query and a set of bound parameters:

```python
>>> print(compiled_stmt.string)
SELECT orders.id, orders.item_type, orders.status, orders.made_by, orders.timestamp, orders.item_details 
FROM orders 
WHERE orders.status = %(status_1)s AND orders.item_type = %(item_type_1)s
>>> compiled_stmt.params
{'status_1': <Status.InProgress: 2>,
 'item_type_1': <KrabbyPattyItemType.Special: 4>}
```

It doesn't matter if we use `orders.c.item_type == item_type` or `== KrabbyPattyItemType.Special`,
from SA's `__eq__`'s point-of-view, the right-hand-side is just some variable which it adds a new parameter to the query.

And while it _can_ be compiled so that the parameters are replaced:

```python
>>> print(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
SELECT orders.id, orders.item_type, orders.status, orders.made_by, orders.timestamp, orders.item_details 
FROM orders 
WHERE orders.status = 'InProgress' AND orders.item_type = 'Special'
```

By default, it does not - and for a good reason!

Consider our example: the DB needs to do a considerable amount of work for each new query - parsing the SQL, analyzing and building a query plan, and executing the plan.
But the first two are a prime target for caching! If we were to inline all the parameters into the query's text, we'll miss out on a substantial optimization.
In our example, we only ever produce 3 different queries, but there are many high-cardinality values we tend to pass as SQL parameters like user IDs, dates, limits and so on.

In short, it's a good idea for SA to _not_ mix the query and parameters before passing them to the _driver_.

In this case, the parametrized version of the query is passed to `psycopg` which manages the DB side of things,
which in turn knows that because this query is parametrized, there's a good chance we are going to use it many more times.

`psycopg` has a pretty standard [strategy][psycopg-prepared] for giving you more performance: 

> A query is prepared automatically after it is executed more than `prepare_threshold` [Default value: 5] times on a connection.

[psycopg-prepared]: https://www.psycopg.org/psycopg3/docs/advanced/prepare.html#prepared-statements

Let's see it in action. We can query all the [prepared statements][pg_prepared_statements] in the current session using:

[pg_prepared_statements]: https://www.postgresql.org/docs/current/view-pg-prepared-statements.html

```sql
SELECT * FROM pg_prepared_statements
```

It'll start out empty, and if we execute our query once it will stay empty:

```python
>>> query_latest_in_prog(conn, item_type=KrabbyPattyItemType.Special)
[(4400001, ..., {'name': 'Krusty Krab Pizza', 'ingredients': [...]}),
 (4400002, ..., {'name': 'Triple Krabby Supreme', 'ingredients': [...]})]
>>> conn.execute(text("SELECT * FROM pg_prepared_statements")).mappings().all()
[]
```

But after 5 more times:

```python
for _ in range(5): query_latest_in_prog(conn, item_type=KrabbyPattyItemType.Special)
```

We'll get back something like this:

```python
>>> conn.execute(text("SELECT * FROM pg_prepared_statements")).mappings().all()[0]
{'name': '_pg3_1', 'statement': 'SELECT orders.id, orders.item_type, orders.status, orders.made_by, orders.timestamp, orders.item_details, employee.email_address \nFROM orders JOIN employee ON employee.id = orders.made_by \nWHERE orders.status = $1 AND orders.item_type = $2 ORDER BY orders.timestamp DESC \n LIMIT $3::INTEGER', ..., 'generic_plans': 0, 'custom_plans': 1}
```

The important part here is that the `statement` part has these replacements `WHERE orders.status = $1 AND orders.item_type = $2` which means it will be shared between all the queries that use this statement,
regardless of what the parameters used in a specific invocation, which mean that parsing the actual SQL is only done a fixed number of times. But what about calculating the query plan?

This is captured in the `{ .., 'generic_plans': 0, 'custom_plans': 1 }` part. According to the [PREPARE docs][pg-docs-prepare]:

[pg-docs-prepare]: https://www.postgresql.org/docs/current/sql-prepare.html

> A prepared statement can be executed with either a generic plan or a custom plan.
> A generic plan is the same across all executions, while a custom plan is generated for a specific execution using the parameter values given in that call. 
> 
> Use of a generic plan avoids planning overhead, but in some situations a custom plan will be much more efficient ...

You can see where this is going, right?

Now: a clarification. PostgreSQL is doing a good job here, so we need to choose different `item_type`s to trigger this based on whether or not we created the index in this session after running some queries.
In the first part, we did everything serially in the same session which worked out OK, but if we want to re-recreate this issue in a new session we need to give the query planner something else.

```python
>>> for _ in range(9): query_latest_in_prog(conn, item_type=KrabbyPattyItemType.KelpRings) # Nobody orders Kelp Rings :(
>>> conn.execute(text("SELECT * FROM pg_prepared_statements")).mappings().all()[0]
{'name': '_pg3_1', ..., 'generic_plans': 11, 'custom_plans': 5}
```

Now the query planned is sufficiently confused - it thinks the generic plans are oh-so-much-better, and it ends up passimising our poor specials page:  XXX use correct word XXX

```python
%%timeit -n 5
query_latest_in_prog(conn, item_type=KrabbyPattyItemType.Special)
# 236 ms ± 1.87 ms per loop
```

The quickest fix is to switch 

By default, the `plan_cache_mode` is set to `auto`, so

> [The] server will automatically choose whether to use a generic or custom plan for a prepared statement that has parameters. 
> The current rule for this is that the first five executions are done with custom plans and the average estimated cost of those plans is calculated. 
> Then a generic plan is created and its estimated cost is compared to the average custom-plan cost. 
> **Subsequent executions use the generic plan if its cost is not so much higher than the average custom-plan cost as to make repeated replanning seem preferable.**

```sql
SET LOCAL plan_cache_mode = force_custom_plan
```

```python
%%timeit -n 50
query_latest_in_prog(conn, item_type=KrabbyPattyItemType.Special)
# 1.01 ms ± 151 μs per loop
```

```python
>>> conn.execute(text("SELECT * FROM pg_prepared_statements")).mappings().all()[0]
{'name': '_pg3_1', ..., 'generic_plans': 30, 'custom_plans': 390}
```

## Summary

Typically, when we use code to build SQL we end up with parametrized SQL queries like:

```sql
SELECT * FROM orders WHERE item_type = %s
```

PostgreSQL drivers, like `psycopg`, will [convert][psycopg-prepared] repeated queries into [prepared statements][pg-docs-prepare].
By default, PostgreSQL will compute both a _custom plan_ (which *is* specific to the parameter values) and a _generic plan_ (which is not, and so can be used across all executions),
and will use the generic one if the gains vs. the custom one are estimated to be small enough.

This can effect whether or not a [_partial index_][pg-docs-partial-idx] like:

```sql
CREATE INDEX ix_timestamp_item_type_special ON orders (timestamp)
WHERE item_type = 'Special'
```

will be used.

Because the generic vs custom cost analysis is done on whatever queries happen to execute first in a session,
it's possible (albeit unlikely) that a query will end up using a generic query plan because for some parameters the difference is small enough. But *if a generic plan is selected*, a partial index that depends on specific parameter values will not be used!

So, either:

1. Use `SET LOCAL plan_cache_mode = force_custom_plan` to avoid generic plans, which will "waste" more resources on planning but avoid this issue.
2. Inline values that should always effect query plans instead of passing them as query parameters.
