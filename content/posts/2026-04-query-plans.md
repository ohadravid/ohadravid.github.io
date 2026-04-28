
---
title: "The Best (Query) Plans of Mice and Men"
date: 2026-04-28T13:00:00+00:00
tags: ["databases", "performance", "python"]
type: post
showTableOfContents: true
image: ""
weight: 1
---
<style>
.post .post-content {
    margin-top: 0px;
}

/* We have more code but it's simpler than usual. */
pre code {
    font-size: .75em;
}
</style>
Or rather, of elephants 🐘 and men.

At [$work](https://www.wiz.io/blog/google-closes-deal-to-acquire-wiz), we use PostgreSQL _a lot_. \
So much in fact, that we tend to be ~paranoid~ frugal when _adding even more stuff_ to it.

In this case, we wanted to **add a new index** to speed up a query (with an eye-watering p99 latency of several seconds).
We have a single, big table that is shared between a few different features, but we are only interested in a tiny subset of rows from that table:

```go
// Less than 0.1% of rows match this.
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

TL;DR - Surprisingly, yes! That is because PostgreSQL, under [_certain conditions_][pg-docs-plan-cache-mode], will calculate and reuse query plans *without considering the specific parameters* being used, which could defeat the point of our clever index.

[pg-docs-plan-cache-mode]: https://www.postgresql.org/docs/current/runtime-config-query.html#GUC-PLAN-CACHE-MODE

To understand why, we'll first need to trigger this behavior - which requires setting up a schema with some data, queries, and indexes (or you can jump straight [to the summary](#summary)). 

Now is a good time for a fresh cup of tea! 🍵

## Setting Up Our Tables and Data

For our example, we will model an order management app for the famous "Krusty Krab" restaurant.
The gist is:
1. We'll have an **_orders_** table with ~4.5M rows, most will be `KrabbyPatty`s and `CoralBites`.
2. We'll have 2 rows with `Special` items.
3. We'll also have an **_employee_** table we need to `JOIN` with.

You can read the full code in the [end](#appendix-a---generating-the-data), but the result is a schema with a few employees, 
a ton of orders - but with a skewed distribution:

```sql
postgres=# SELECT name, email_address FROM employee;
   name    |       email_address
-----------+---------------------------
 SpongeBob | spongebob@bikinibottom.io
 Squidward | squidward@bikinibottom.io
 Mr. Krabs | mrkrabs@bikinibottom.io
(3 rows)
postgres=# SELECT item_type, status, count(*) FROM orders                                                                        GROUP BY item_type, status;
  item_type  |   status   |  count
-------------+------------+---------
 KrabbyPatty | InProgress | 1100000
 KrabbyPatty | Done       | 1100000
 CoralBites  | InProgress | 1100000
 CoralBites  | Done       | 1100000
 Special     | InProgress |       2
(5 rows)
```

_Side note: yes I bought [bikinibottom.io](https://bikinibottom.io/) while writing this blog post. It could not have been helped._

Our example will focus on the few `Special` items, which will be the first two items from the [List of Krabby Patty variations] in _Encyclopedia SpongeBobia_:
- Krusty Krab Pizza
- Triple Krabby Supreme

With all this data in place, we can now build our queries and trigger some interesting PostgreSQL behaviors.

<p style="text-align:center;">
    <img src="/2026-04-query-plans/Just_One_Bite_165.webp" 
        alt="Squidward looking into the Krabby Patty vault in the Just One Bite episode" loading="lazy" 
        style="width: min(100%, max(66%, 420px));" width="1422" height="1080" />
</p>


## Our Query Function

Let's switch from Go with GORM to Python with SQLAlchemy, using the [Core API][insert-docs]. We do not need a full ORM for this example, and ORMs are not _in vogue_ anyway. Again, the full code for the schema can be found at the [end](#appendix-a---generating-the-data).

We have 3 queries we want to perform against these tables - think of them as separate pages in our "Krusty Krab App":

1. Query the latest `InProgress` orders, regardless of `item_type`.
2. Query the latest `InProgress` orders for `KrabbyPatty` items.
3. Query the latest `InProgress` orders for `Special` items.

This third use case represents our original problem: as we'll see, 
this query will be slower, and adding a full index is wasteful because we only care about a fraction of the rows in the table.

Using SQLAlchemy's [`select()`][select-docs], we can implement all 3 in a single simple Python function:

[select-docs]: https://docs.sqlalchemy.org/en/21/tutorial/data_select.html#the-select-sql-expression-construct

```python
def query(conn, item_type: Optional[KrabbyPattyItemType] = None, limit: Optional[int] = 100):
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

Now, 4.5 million rows isn't _that_ much, and computers are really fast. So, let's spin up a `jupyter lab` [notebook] and quickly measure[^1]:

[notebook]: https://github.com/ohadravid/ohadravid.github.io/blob/main/static/2026-04-query-plans/code/query_plans_demo.ipynb

[^1]: The `%%timeit -n` syntax tells the notebook to run the cell multiple times in a loop and measure the average time it took across multiple runs. In this case, 7 runs with 5 loops each.

```python
%%timeit -n 5
query(conn)
# 124 ms ± 2.27 ms per loop
```

Big oof. Let's create a simple, regular index to speed this up:

```sql
CREATE INDEX ix_status_ts ON orders (status, timestamp)
```

The table takes about 200MB of storage and this new index requires an additional 100MB[^4], but as you might expect:

[^4]: We can measure the sizes using:

    ```sql
    postgres=# SELECT relname, pg_size_pretty(pg_relation_size(relid)) FROM pg_catalog.pg_statio_user_tables;
     relname  | pg_size_pretty
    ----------+----------------
     employee | 8192 bytes
     orders   | 219 MB
    (2 rows)
    postgres=# SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)) FROM pg_stat_all_indexes WHERE indexrelname LIKE 'ix%';
              indexrelname          | pg_size_pretty
    --------------------------------+----------------
     ix_status_ts                   | 104 MB
     ix_timestamp_item_type_special | 16 kB
    (2 rows)
    ```

```python
%%timeit -n 5
query(conn)
# 1.51 ms ± 208 μs per loop
```

Much better! And filtering for Krabby Patties is almost as fast:

```python
%%timeit -n 5
query(conn, item_type=KrabbyPattyItemType.KrabbyPatty)
# 1.6 ms ± 141 μs per loop 
```

But... our poor `Special`s page!

```python
%%timeit -n 5
query(conn, item_type=KrabbyPattyItemType.Special)
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

which is equivalent (or is it?) to a regular `WHERE status = 'InProgress' AND orders.item_type = 'Special'`.

We want it to be faster, but without paying for a full index.

So let's create a [partial index][pg-docs-partial-idx]: we'll still use the `timestamp` column, 
but add a filter so that only `Special` items are included:

[pg-docs-partial-idx]: https://www.postgresql.org/docs/current/indexes-partial.html

```sql
CREATE INDEX ix_timestamp_item_type_special ON orders (timestamp)
WHERE item_type = 'Special'
```

This new index is a few KB in size - much better than the 132MB a full `(item_type, timestamp)` index would consume - and as expected, doesn't affect the other queries:

```python
%%timeit -n 10
query(conn, item_type=KrabbyPattyItemType.KrabbyPatty)
# 2.14 ms ± 401 μs per loop
```

But is our query fast now?

```python
%%timeit -n 5
query(conn, item_type=KrabbyPattyItemType.Special)
# 238 ms ± 2.41 ms per loop
```

Not at all! Huh? Maybe we didn't set up the index correctly? Let's double-check using a raw query:

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

What? It _is_ faster? But... isn't this the same query?

<p style="text-align:center;">
    <img src="/2026-04-query-plans/corporate_wants_you_to_find.webp" 
        alt="corporate needs you to tell the difference meme" loading="lazy" 
        style="width: min(100%, max(66%, 420px));" width="1360" height="762" />
</p>


## But Why?

So, what is happening here? To find out, let's take this SQL statement 

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

What we get back is perhaps not exactly what you might expect. We get back a _parameterized_ query and a set of bound parameters:

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
from SA's `__eq__` point-of-view, the right-hand side is an opaque variable which it adds as a new parameter to the query.

And while it _can_ be compiled so that the parameters are replaced:

```python
>>> print(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
SELECT orders.id, orders.item_type, orders.status, orders.made_by, orders.timestamp, orders.item_details 
FROM orders 
WHERE orders.status = 'InProgress' AND orders.item_type = 'Special'
```

By default, it does not - and for a good reason!

Consider our example: the DB needs to do considerable work for each new query:

1. Parsing the SQL.
2. Analyzing and building a query plan.
3. Executing the plan.

But the first two are prime targets for caching! If we were to inline all the parameters into the query's text, we'll miss out on an important optimization.
In our example, we only ever produce 3 different queries, but there are many high-cardinality values we tend to pass as SQL parameters like user IDs, dates, limits and so on.

In short, it's a good idea for SA to _not_ mix the query and parameters before passing them to the _driver_.

In this case, the parameterized version of the query is passed to `psycopg` which manages the DB side of things.
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
>>> query(conn, item_type=KrabbyPattyItemType.Special)
[(4400001, ..., {'name': 'Krusty Krab Pizza', 'ingredients': [...]}),
 (4400002, ..., {'name': 'Triple Krabby Supreme', 'ingredients': [...]})]
>>> conn.execute(text("SELECT * FROM pg_prepared_statements")).mappings().all()
[]
```

But after 5 more times:

```python
for _ in range(5): query(conn, item_type=KrabbyPattyItemType.Special)
```

We'll get back something like this:


<style>
/* Highlighted code samples have a black background for some reason  */
.highlighted-code > pre > code {
    background-color: unset;
}
</style>

```python {hl_lines=[6] class="highlighted-code"}
>>> conn.execute(text("SELECT * FROM pg_prepared_statements")).mappings().all()
[{
  'name': '_pg3_1', 
  'statement': """SELECT orders.id, orders.item_type, orders.status, orders.made_by, orders.timestamp, orders.item_details, employee.email_address
    FROM orders JOIN employee ON employee.id = orders.made_by
    WHERE orders.status = $1 AND orders.item_type = $2 ORDER BY orders.timestamp DESC 
    LIMIT $3::INTEGER""", 
  ..., 
  'generic_plans': 0, 
  'custom_plans': 1
}]
```

The important part here is that the `statement` is a parametrized SQL query with these replacements (`item_type = $1`), meaning all our queries that include both filters (and limit) will resolve into this prepared statement,
regardless of the parameters used in a specific invocation.

The immediate benefit here is that **parsing** the SQL is only done a fixed number of times. But what about calculating the query plan?

This is captured in the `{ .., 'generic_plans': 0, 'custom_plans': 1 }` part. According to the [PREPARE docs][pg-docs-prepare]:

[pg-docs-prepare]: https://www.postgresql.org/docs/current/sql-prepare.html

> A prepared statement can be executed with either a generic plan or a custom plan.
> A generic plan is the same across all executions, while a custom plan is generated for a specific execution using the parameter values given in that call. 
> 
> Use of a generic plan avoids planning overhead, but in some situations a custom plan will be much more efficient ...

You can see where this is going, right?

A small clarification: PostgreSQL is generally doing a decent job here, 
while we try to fool the query planer to trigger a specific issue.
In the first part, we did everything serially in the same session, but to re-recreate this issue in a new session that already includes the new index, we need to give the query planner something else to chew on.

```python {hl_lines=[6,7] class="highlighted-code"}
>>> # Nobody orders Kelp Rings :(
>>> for _ in range(9): query(conn, item_type=KrabbyPattyItemType.KelpRings) 
>>> conn.execute(text("SELECT * FROM pg_prepared_statements")).mappings().all()[0]
{
  'name': '_pg3_1',
  'generic_plans': 11,
  'custom_plans': 5,
  ...,
}
```

Now `custom_plans` is stuck at 5, and _only_`generic_plans` is ticking up. So how does the DB decide between a custom and a generic plan?

By default, the `plan_cache_mode` is set to `auto`, in which:

> [The] server will automatically choose .. 
> the **first five executions** are done with custom plans and the average estimated cost of those plans is calculated. 
> Then a generic plan is created and its estimated cost is compared to the average custom-plan cost. 
> **Subsequent executions use the generic plan if its cost is not so much higher than the average custom-plan cost as to make repeated replanning seem preferable.**

Essentially, we got the query planner sufficiently confused - it thinks the generic plan looks oh-so-much-better than the custom one because for `KelpRings` there isn't any difference.

But, when it's finally time for our poor `Special`s page, PostgreSQL sees the same parametrized statement, checks the records which shows that generic plans are worth it, and uses that to run the query - which is terribly slow:

```python
%%timeit -n 5
query(conn, item_type=KrabbyPattyItemType.Special)
# 236 ms ± 1.87 ms per loop
```

We can also confirm by using `EXPLAIN` - note the `$2` here:

```sql
EXPLAIN EXECUTE _pg3_1('InProgress', 'Special', 100); 
-- ->  Index Scan Backward using ix_status_ts on orders ..
--       .. snip ..
--       Filter: (item_type = $2)
```

The quickest fix is to switch `plan_cache_mode` to `force_custom_plan`.

```sql
SET LOCAL plan_cache_mode = force_custom_plan
```

By "wasting" more resources on re-planning each time the query is executing we get a plan that uses our partial index, and results in a fast execution:

```python
%%timeit -n 50
query(conn, item_type=KrabbyPattyItemType.Special)
# 1.01 ms ± 151 μs per loop
```

As expected, now `custom_plans` is ticking up:

```python
>>> conn.execute(text("SELECT * FROM pg_prepared_statements")).mappings().all()[0]
{'generic_plans': 30, 'custom_plans': 390, 'name': '_pg3_1', ..., }
```

And we can confirm with `EXPLAIN` as well - we see our index being used, and instead of `$1` we see the `'InProgress'` status:

```sql
EXPLAIN EXECUTE _pg3_1('InProgress', 'Special', 100); 
-- -> Index Scan Backward using ix_timestamp_item_type_special on orders ..
--       .. snip ..
--       Filter: (status = 'InProgress'::status)
```

Victory!

We could also add a literal `AND orders.item_type = 'Special'` clause to the query,
or try to be clever with SA:

```python
# bad bad bad
orders.c.item_type == text(repr(KrabbyPattyItemType.Special.name))
```

I don't recommend it, but also it's kind of charming? You get to decide.

Anyway! A victory indeed.

## Summary

Typically, when we use code to build SQL we end up with parametrized SQL queries like:

```sql
SELECT * FROM orders
WHERE item_type = %s
```

PostgreSQL drivers, like `psycopg`, will [convert][psycopg-prepared] repeated queries into [prepared statements][pg-docs-prepare].
By default, PostgreSQL will compute both a _custom plan_ (which *is* specific to the parameter values) and a _generic plan_ (which is not, and so can be used across all executions),
and will use the generic one if the gains vs. the custom one are estimated to be small enough.

This can affect whether or not a [_partial index_][pg-docs-partial-idx] like:

```sql
CREATE INDEX ix_timestamp_item_type_special ON orders (timestamp)
WHERE item_type = 'Special'
```

will be used.

Because the generic vs custom cost[^5] analysis is done on whatever queries happen to execute first in a session,
it's possible (albeit unlikely) that a query will end up using a generic plan because for some parameters the difference is small enough. But *if a generic plan is selected*, a partial index that depends on specific parameter values will not be used!

[^5]: 
    We have used the default PostgreSQL settings, but even switching to the more-correct-for-SSDs `random_page_cost` value:
    
    ```sql
    SET random_page_cost=1.1
    ```

    Will still select the generic plans in this case.


So, either:

1. Use `SET LOCAL plan_cache_mode = force_custom_plan` to avoid generic plans, which will "waste" more resources on planning but avoid this issue.
2. Inline values that should always affect query plans instead of passing them as query parameters.

--

_Discuss on [lobste.rs](https://lobste.rs/s/a7aoor/best_query_plans_mice_men), [HN](https://news.ycombinator.com/item?id=47936051)._


_If you liked this, you might also like [BTrees, Inverted Indices, and a Model for Full Text Search]({{< ref "/posts/2025-04-08-btrees-and-mental-models.md" >}}) and [Why is calling my asm function from Rust slower than calling it from C?]({{< ref "2025-12-rav1d-faster-asm.md" >}})._

## Appendix A - Generating the Data

We'll be using SQLAlchemy to define the schema and generate all the data we need (you can checkout the [full code on GitHub](https://github.com/ohadravid/ohadravid.github.io/blob/main/static/2026-04-query-plans/code/)).

We'll use 2 tables. A small employee table:

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

Into this table we can [`insert()`][insert-docs] a few chosen employees:

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

And a soon-to-be-huge `orders`[^2] table:

[^2]: Yes, using plural form is worse _and_ inconsistent with the `employee` table, but we don't want to use `order` to avoid confusion with `ORDER BY`.


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

Now we can fill it with 22,000[^3] Krabby Patties and Coral Bites (times 100), spread over a few years - some orders will be `InProgress`, and some will be `Done`:

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

Aaaand done! We now have all the data we need - but we also need to query this data.

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